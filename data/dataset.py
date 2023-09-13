import os
from os import path
import sys

import scipy.special

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tableshift"))
from collections import Counter

import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F

import openml
from openml import tasks, runs
from data.utils.util_functions import load_opt
from datasets import load_dataset
from tableshift import get_dataset, get_iid_dataset
from utils import utils



class Dataset():
    def __init__(self, args, logger):
        self.logger = logger
        if args.benchmark in ["openml-cc18", "tableshift", "shifts", "folktables", "openml-regression", "scikit-learn"]:
            benchmark = args.benchmark.replace("-", "_")
            (train_x, valid_x, test_x), (train_y, valid_y, test_y), cat_indices, regression = eval(f"self.get_{benchmark}_dataset")(args)
        else:
            raise NotImplementedError

        if float(args.train_ratio) < 1:
            train_x = train_x.iloc[:int(len(train_x) * float(args.train_ratio)), :]
            train_y = train_y.iloc[:int(len(train_y) * float(args.train_ratio)), :]

        if args.smote:
            from imblearn.over_sampling import SMOTENC
            train_x, train_y = SMOTENC(categorical_features=cat_indices, random_state=args.seed).fit_resample(train_x, train_y)

        ##### preprocessing #####
        cont_indices = np.array(sorted(set(np.arange(train_x.shape[-1])).difference(set(cat_indices))))
        self.emb_dim = []
        if len(cont_indices):
            self.input_scaler = getattr(sklearn.preprocessing, args.normalizer)()
            self.input_scaler.fit(np.concatenate([train_x.iloc[:, cont_indices], valid_x.iloc[:, cont_indices]], axis=0))
            train_cont_x = self.input_scaler.transform(train_x.iloc[:, cont_indices])
            valid_cont_x = self.input_scaler.transform(valid_x.iloc[:, cont_indices])
            if not args.benchmark in ["openml-cc18", "openml-regression", "scikit-learn"]: # TODO (important!) add new synthetic corruption benchmarks
                args.shift_type = None
            test_cont_x, test_cont_mask_x = Dataset.get_corrupted_data(
                np.array(test_x.iloc[:, cont_indices]),
                np.array(train_x.iloc[:, cont_indices]),
                data_type="numerical",
                shift_type=args.shift_type,
                shift_severity=args.shift_severity,
                imputation_method=args.missing_imputation_method,
            )
            test_cont_x = self.input_scaler.transform(test_cont_x)
        else:
            train_cont_x, test_cont_mask_x, valid_cont_x, test_cont_x = np.array([]), np.array([]), np.array([]), np.array([])

        if len(cat_indices):
            self.input_one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.input_one_hot_encoder.fit(np.concatenate([train_x.iloc[:, cat_indices], valid_x.iloc[:, cat_indices]], axis=0))
            train_cat_x = self.input_one_hot_encoder.transform(train_x.iloc[:, cat_indices])
            valid_cat_x = self.input_one_hot_encoder.transform(valid_x.iloc[:, cat_indices])
            if not args.benchmark in ["openml-cc18", "openml-regression", "scikit-learn"]: # TODO (important!) add new synthetic corruption benchmarks
                args.shift_type = None
            test_cat_x, test_cat_mask_x = Dataset.get_corrupted_data(
                np.array(test_x.iloc[:, cat_indices]),
                np.array(train_x.iloc[:, cat_indices]),
                data_type="categorical",
                shift_type=args.shift_type,
                shift_severity=args.shift_severity,
                imputation_method=args.missing_imputation_method,
            )
            test_cat_x = self.input_one_hot_encoder.transform(test_cat_x)
            test_cat_mask_x = np.concatenate([np.repeat(test_cat_mask_x[:, category_idx][:, None], len(category), axis=1) for category_idx, category in enumerate(self.input_one_hot_encoder.categories_)], axis=1)
        else:
            train_cat_x, valid_cat_x, test_cat_x, test_cat_mask_x = np.array([]), np.array([]), np.array([]), np.array([])

        self.train_x = np.concatenate([train_cont_x if len(cont_indices) else train_cont_x.reshape(train_cat_x.shape[0], 0), train_cat_x if len(cat_indices) else train_cat_x.reshape(train_cont_x.shape[0], 0)], axis=-1)
        self.valid_x = np.concatenate([valid_cont_x if len(cont_indices) else valid_cont_x.reshape(valid_cat_x.shape[0], 0), valid_cat_x if len(cat_indices) else valid_cat_x.reshape(valid_cont_x.shape[0], 0)], axis=-1)
        self.test_x = np.concatenate([test_cont_x if len(cont_indices) else test_cont_x.reshape(test_cat_x.shape[0], 0), test_cat_x if len(cat_indices) else test_cat_x.reshape(test_cont_x.shape[0], 0)], axis=-1)
        self.test_mask_x = np.concatenate([test_cont_mask_x if len(cont_indices) else test_cont_mask_x.reshape(test_cat_mask_x.shape[0], 0), test_cat_mask_x if len(cat_indices) else test_cat_mask_x.reshape(test_cont_mask_x.shape[0], 0)], axis=-1)

        if regression:
            self.output_scaler = getattr(sklearn.preprocessing, args.normalizer)()
            self.output_scaler.fit(np.concatenate([train_y, valid_y], axis=0))
            self.train_y = self.output_scaler.transform(train_y)
            self.valid_y = self.output_scaler.transform(valid_y)
            self.test_y = self.output_scaler.transform(test_y)
        else:
            self.output_one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.output_one_hot_encoder.fit(np.concatenate([train_y, valid_y], axis=0))
            self.train_y = self.output_one_hot_encoder.transform(train_y)
            self.valid_y = self.output_one_hot_encoder.transform(valid_y)
            self.test_y = self.output_one_hot_encoder.transform(test_y)

        train_data = torch.utils.data.TensorDataset(torch.FloatTensor(self.train_x).type(torch.float32), torch.FloatTensor(self.train_y).type(torch.float32))
        valid_data = torch.utils.data.TensorDataset(torch.FloatTensor(self.valid_x).type(torch.float32), torch.FloatTensor(self.valid_y).type(torch.float32))
        test_data = torch.utils.data.TensorDataset(torch.FloatTensor(self.test_x).type(torch.float32), torch.FloatTensor(self.test_mask_x).type(torch.float32), torch.FloatTensor(self.test_y).type(torch.float32))

        self.in_dim, self.out_dim = self.train_x.shape[-1], self.train_y.shape[-1]
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, worker_init_fn=utils.set_seed_worker, generator=utils.get_generator(args.seed))
        self.valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.train_batch_size, shuffle=False, worker_init_fn=utils.set_seed_worker, generator=utils.get_generator(args.seed))
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, worker_init_fn=utils.set_seed_worker, generator=utils.get_generator(args.seed))

        # for embedding
        self.cont_dim = train_cont_x.shape[-1]
        if hasattr(self, 'input_one_hot_encoder'):
            self.emb_dim_list = [(len(category), min(10, (len(category) + 1) // 2)) if len(cat_indices) else (0, 0) for category in self.input_one_hot_encoder.categories_]
            self.cat_end_indices = np.cumsum([num_category for num_category, _ in self.emb_dim_list])
            self.cat_start_indices = np.concatenate([[0], self.cat_end_indices], axis=0)[:-1]
            self.cat_indices_groups = [list(range(self.cont_dim + cat_start_index, self.cont_dim + cat_end_index)) for cat_start_index, cat_end_index in zip(self.cat_start_indices, self.cat_end_indices)]
        else:
            self.emb_dim_list, self.cat_end_indices, self.cat_start_indices, self.cat_indices_groups = [], np.array([]), np.array([]), []

        # print dataset info
        train_counts = np.unique(np.argmax(self.train_y, axis=1), return_counts=True)
        valid_counts = np.unique(np.argmax(self.valid_y, axis=1), return_counts=True)
        test_counts = np.unique(np.argmax(self.test_y, axis=1), return_counts=True)

        logger.info(f"dataset size | train: {len(self.train_x)}, valid: {len(self.valid_x)}, test: {len(self.test_x)}")
        logger.info(f"Class distribution - train {np.round(train_counts[1] / np.sum(train_counts[1]), 2)}, {train_counts}")
        logger.info(f"Class distribution - valid {np.round(valid_counts[1] / np.sum(valid_counts[1]), 2)}, {valid_counts}")
        logger.info(f"Class distribution - test {np.round(test_counts[1] / np.sum(test_counts[1]), 2)}, {test_counts}")


    def get_openml_cc18_dataset(self, args):
        benchmark_list_path = "data/OpenML-CC18/benchmark_list.csv"
        if not os.path.exists(benchmark_list_path):
            benchmark = openml.study.get_suite("OpenML-CC18")
            benchmark_df = tasks.list_tasks(output_format="dataframe", task_id=benchmark.tasks)
            benchmark_df.to_csv(benchmark_list_path)
        else:
            benchmark_df = pd.read_csv(benchmark_list_path)
        target_feature = benchmark_df[benchmark_df["name"] == args.dataset].iloc[0]["target_feature"]

        dataset = openml.datasets.get_dataset(args.dataset)
        x, y, cat_indicator, _ = dataset.get_data(target=target_feature, dataset_format="dataframe")
        y = pd.DataFrame(y)
        cat_indices = np.argwhere(np.array(cat_indicator) == True).flatten()
        regression = False

        if args.shift_type in ["numerical", "categorical"]:
            if (len(cat_indices) == x.shape[-1] and args.shift_type == "numerical") or (len(cat_indices) == 0 and args.shift_type == "categorical"):
                raise Exception(f'No {args.shift_type} columns in {args.dataset} dataset!')

            train_indices, test_indices = self.split_dataset_by_natural_shift(x, y, cat_indices, args.shift_type, args.shift_severity, regression=False)
            train_x, train_y = x.iloc[train_indices, :], y.iloc[train_indices, :]
            test_x, test_y = x.iloc[test_indices, :], y.iloc[test_indices, :]
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.25, random_state=42)
        else:
            train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4, random_state=42)
            valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y, test_size=0.5, random_state=42)
        return (train_x, valid_x, test_x), (train_y, valid_y, test_y), cat_indices, regression


    def get_tableshift_dataset(self, args):
        from tableshift.core.features import PreprocessorConfig, get_categorical_columns
        dataset = get_dataset(
            args.dataset,
            cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tableshift/tableshift/tmp"),
            preprocessor_config=PreprocessorConfig(categorical_features="passthrough", numeric_features="passthrough")
        )
        self.logger.info(f'out-of-distribution dataset') if dataset.is_domain_split else self.logger.info(f'in-distribution dataset')
        regression = False

        train_x, train_y, _, _ = dataset.get_pandas("train")
        valid_x, valid_y, _, _ = dataset.get_pandas("validation")
        test_x, test_y, _, _ = dataset.get_pandas("ood_test") if dataset.is_domain_split else dataset.get_pandas("test")
        cat_indices = np.array(sorted([train_x.columns.get_loc(c) for c in get_categorical_columns(train_x)]))
        return (train_x, valid_x, test_x), (pd.DataFrame(train_y), pd.DataFrame(valid_y), pd.DataFrame(test_y)), cat_indices, regression


    def get_shifts_dataset(self, args):
        dataset_path_name = 'weather' if 'weather' in args.dataset else args.dataset
        regression = True if args.dataset in ['weather_reg', 'power'] else False
        cat_indices = np.array([])

        if (not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"shifts/{dataset_path_name}/train.csv"))) and 'weather' in args.dataset:
            csv_list = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), f"shifts/{dataset_path_name}/shifts_canonical_train.csv"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), f"shifts/{dataset_path_name}/shifts_canonical_dev_in.csv"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), f"shifts/{dataset_path_name}/shifts_canonical_eval_in.csv"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), f"shifts/{dataset_path_name}/shifts_canonical_eval_out.csv"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), f"shifts/{dataset_path_name}/shifts_canonical_dev_out.csv"),
            ]
            train_df, valid_df, test_df = self.concat_csvs(csv_list, args)
        else:
            train_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"shifts/{dataset_path_name}/train.csv")).dropna()
            valid_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"shifts/{dataset_path_name}/dev_in.csv")).dropna()
            test_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"shifts/{dataset_path_name}/dev_out.csv")).dropna()

        if args.dataset == 'weather_reg':
            train_x = train_df[train_df.columns.difference(['fact_temperature', 'climate'])].astype(np.float32)
            valid_x = valid_df[train_df.columns.difference(['fact_temperature', 'climate'])].astype(np.float32)
            test_x = test_df[train_df.columns.difference(['fact_temperature', 'climate'])].astype(np.float32)
            train_y = train_df[['fact_temperature']].astype(np.float32)
            valid_y = valid_df[['fact_temperature']].astype(np.float32)
            test_y = test_df[['fact_temperature']].astype(np.float32)
        elif args.dataset == 'weather_cls':
            train_x = train_df[train_df.columns.difference(['fact_cwsm_class', 'climate'])].astype(np.float32)
            valid_x = valid_df[train_df.columns.difference(['fact_cwsm_class', 'climate'])].astype(np.float32)
            test_x = test_df[train_df.columns.difference(['fact_cwsm_class', 'climate'])].astype(np.float32)
            train_y = train_df[['fact_cwsm_class']].astype(np.float32)
            valid_y = valid_df[['fact_cwsm_class']].astype(np.float32)
            test_y = test_df[['fact_cwsm_class']].astype(np.float32)
        elif args.dataset == 'power':
            train_x = train_df[train_df.columns.difference(['power'])].astype(np.float32)
            valid_x = valid_df[train_df.columns.difference(['power'])].astype(np.float32)
            test_x = test_df[train_df.columns.difference(['power'])].astype(np.float32)
            train_y = train_df[['power']].astype(np.float32)
            valid_y = valid_df[['power']].astype(np.float32)
        return (train_x, valid_x, test_x), (train_y, valid_y, test_y), cat_indices, regression


    def get_folktables_dataset(self, args):
        from folktables import ACSDataSource, ACSIncome, ACSPublicCoverage
        cat_indices = np.array([])
        regression = False

        if args.dataset == 'state':
            data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
            ca_data = data_source.get_data(states=["CA"], download=True)
            mi_data = data_source.get_data(states=["MI"], download=True)
            train_x, train_y, _ = ACSIncome.df_to_numpy(ca_data)
            test_x, test_y, _ = ACSIncome.df_to_numpy(mi_data)
            train_x, train_y, test_x, test_y = pd.DataFrame(train_x), pd.DataFrame(train_y), pd.DataFrame(test_x), pd.DataFrame(test_y)
        elif args.dataset in ['time', 'state_time', 'time_state']:
            train_x_list, train_y_list, test_x_list, test_y_list = [], [], [], []
            for year in [2014, 2015, 2016]: # train
                data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
                acs_data = data_source.get_data(states=["CA"], download=True)
                features, labels, _ = ACSPublicCoverage.df_to_numpy(acs_data)
                train_x_list.append(features)
                train_y_list.append(labels)
            for year in [2017, 2018]: # test
                data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
                acs_data = data_source.get_data(states=["CA" if args.dataset == "time" else "MI"], download=True)
                features, labels, _ = ACSPublicCoverage.df_to_numpy(acs_data)
                test_x_list.append(features)
                test_y_list.append(labels)
            train_x = pd.DataFrame(np.concatenate(train_x_list, axis=0))
            train_y = pd.DataFrame(np.concatenate(train_y_list, axis=0))
            test_x = pd.DataFrame(np.concatenate(test_x_list, axis=0))
            test_y = pd.DataFrame(np.concatenate(test_y_list, axis=0))
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.25, random_state=42)
        print(f"train_y: {train_y}" )
        return (train_x, valid_x, test_x), (train_y, valid_y, test_y), cat_indices, regression


    def get_openml_regression_dataset(self, args):
        from scipy.io import arff
        config = load_opt(args.dataset)
        regression = True
        if args.dataset in ["cholesterol", "sarcos", "boston", "news"]:
            data = arff.loadarff(config['path'])
            df = pd.DataFrame(data[0])

            if args.dataset not in ['sarcos', 'news']:
                str_df = df.select_dtypes([object]) # load as dataframe and convert datatypes to float
                str_df = str_df.stack().str.decode('utf-8').unstack()
                str_df.replace(to_replace='?', value=np.nan, inplace=True)
                df[str_df.columns] = str_df.astype(np.float32)

            dataset_df = df.dropna()
            wo_target = list(dataset_df.columns[:-1])
            target = list([dataset_df.columns[-1]])
            x, y = dataset_df[wo_target], dataset_df[target]
        elif args.dataset in ["abalone", "seattlecrime6", "diamonds", "Brazilian_houses", "topo_2_1", "house_sales", "particulate-matter-ukair-2017", "analcatdata_supreme", "delays_zurich_transport", "Bike_Sharing_Demand", "nyc-taxi-green-dec-2016", "visualizing_soil", "SGEMM_GPU_kernel_performance"]:
            dataset = load_dataset("inria-soda/tabular-benchmark", data_files=f"reg_cat/{args.dataset}.csv")
            column_names = dataset['train'].column_names
            wo_target = dataset['train'].column_names[:-1]
            target = [dataset['train'].column_names[-1]]

            pandas_dataset = dataset['train'].to_pandas()
            x, y = pandas_dataset[wo_target], pandas_dataset[target]
        else:
            raise NotImplementedError

        cat_indicator = [True if column_name in config['nominal_columns'] else False for column_name in x.columns]
        cat_indices = np.argwhere(np.array(cat_indicator) == True).flatten()
        if args.shift_type in ["numerical", "categorical"]:
            if (len(cat_indices) == x.shape[-1] and args.shift_type == "numerical") or (len(cat_indices) == 0 and args.shift_type == "categorical"):
                raise Exception(f'No {args.shift_type} columns in {args.dataset} dataset!')

            train_indices, test_indices = self.split_dataset_by_natural_shift(x, y, cat_indices, args.shift_type, args.shift_severity, regression=True)
            train_x, train_y = x.iloc[train_indices, :], y.iloc[train_indices, :]
            test_x, test_y = x.iloc[test_indices, :], y.iloc[test_indices, :]
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.25, random_state=42)
        else:
            train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4, random_state=42)
            valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y, test_size=0.5, random_state=42)
        return (train_x, valid_x, test_x), (train_y, valid_y, test_y), cat_indices, regression


    def get_scikit_learn_dataset(self, args):
        from sklearn import datasets
        regression = False
        if args.dataset == 'make_classification':
            n_features = 30 # number of independent features
            n_informative = 5 # number of informative features
            class_sep = 1 # default 1, where larger value makes classification easier
            n_redundant = n_features - n_informative # number of informative features
            x, y = sklearn.datasets.make_classification(n_samples=5000, class_sep=class_sep, n_classes=10, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, random_state=args.seed, shuffle=True)
        elif args.dataset == 'two_moons':
            x, y = sklearn.datasets.make_moons(n_samples=5000, random_state=args.seed, noise=0.3, shuffle=True) # noise = amount of noise added to moons dataset
        else:
            raise NotImplementedError
        x, y = pd.DataFrame(x), pd.DataFrame(y)
        cat_indices = np.array([])
        if args.shift_type in ["numerical", "categorical"]:
            if (len(cat_indices) == x.shape[-1] and args.shift_type == "numerical") or (len(cat_indices) == 0 and args.shift_type == "categorical"):
                raise Exception(f'No {args.shift_type} columns in {args.dataset} dataset!')

            train_indices, test_indices = self.split_dataset_by_natural_shift(x, y, cat_indices, args.shift_type, args.shift_severity, regression=False)
            train_x, train_y = x.iloc[train_indices, :], y.iloc[train_indices, :]
            test_x, test_y = x.iloc[test_indices, :], y.iloc[test_indices, :]
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.25, random_state=42)
        else:
            train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4, random_state=42)
            valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y, test_size=0.5, random_state=42)
        return (train_x, valid_x, test_x), (train_y, valid_y, test_y), cat_indices, regression


    def concat_csvs(self, path_list, args): # for shifts benchmark
        import pandas as pd
        pd_list = []
        for path in path_list:
            pd_list.append(pd.read_csv(path))
        train_pd = pd.concat(pd_list[:3])
        train_pd.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"shifts/weather/train.csv"))
        eval_pd = pd.concat([pd_list[3]])
        eval_pd.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"shifts/weather/dev_in.csv"))
        test_pd = pd.concat(pd_list[4:])
        test_pd.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"shifts/weather/dev_out.csv"))
        return train_pd, eval_pd, test_pd


    @staticmethod
    def get_corrupted_data(test_data, train_data, data_type, shift_type, shift_severity, imputation_method):
        mask = np.ones(test_data.shape, dtype=np.int32)
        if shift_type in ["Gaussian", "uniform"] and data_type == "numerical":
            scaler = StandardScaler()
            scaler.fit(train_data)
            noise = np.random.randn(*test_data.shape) if shift_type == "Gaussian" else np.random.uniform(low=-1, high=1, size=test_data.shape)
            test_data = test_data.astype(np.float32) + shift_severity * noise * np.sqrt(scaler.var_)
        elif shift_type in ["random_drop", "column_drop"]:
            assert 0 <= shift_severity <= 1

            if shift_type == "random_drop":
                mask = (np.random.rand(*test_data.shape) >= shift_severity).astype(np.int32)
            elif shift_type == "column_drop":
                mask = np.repeat((np.random.rand(*test_data.shape[1:]) >= shift_severity).astype(np.int32)[None, :], test_data.shape[0], axis=0)
            imputed_data = Dataset.get_imputed_data(test_data, train_data, data_type, imputation_method)

            if not isinstance(test_data, np.ndarray):
                test_data = test_data.detach().cpu().numpy()

            if data_type == "numerical":
                test_data = mask * test_data + (1 - mask) * imputed_data
            elif data_type == "categorical":
                for row_idx in range(test_data.shape[0]):
                    for col_idx in range(test_data.shape[-1]):
                        if mask[row_idx][col_idx] == 0:
                            test_data[row_idx][col_idx] = imputed_data[row_idx][col_idx]
        return test_data, mask


    @staticmethod
    def get_imputed_data(test_data, train_data, data_type, imputation_method):
        if data_type == "numerical":
            if imputation_method == "zero":
                # if isinstance(test_data, torch.Tensor):
                #     test_data = test_data.cpu()
                # imputed_data = np.zeros_like(test_data)
                imputed_data = np.repeat(np.mean(train_data, axis=0)[None, :], len(test_data), axis=0)
            # elif imputation_method == "mean":
            #     imputed_data = np.repeat(np.mean(train_data, axis=0)[None, :], len(test_data), axis=0)
            elif imputation_method == "emd":
                imputed_data = []
                if isinstance(test_data, torch.Tensor):
                    test_data = test_data.cpu()
                for train_col in train_data.T:
                    imputed_data.append(np.random.choice(train_col, len(test_data)))
                imputed_data = np.stack(imputed_data, axis=-1)
        elif data_type == "categorical":
            if imputation_method == "zero":
                imputed_data = []
                for train_col in train_data.T:
                    train_col = set([str(train_col_elem) for train_col_elem in train_col])
                    imputed_data.append(np.array([max(train_col) + "." for _ in range(len(test_data))]))
                imputed_data = np.stack(imputed_data, axis=-1)
            # elif imputation_method == "mean":  # mode (most frequent value) for categorical variable
            #     imputed_data = []
            #     for train_col in train_data.T:
            #         unique, counts = list(Counter(train_col).keys()), list(Counter(train_col).values())
            #         imputed_data.append(np.array([unique[np.argmax(counts)] for _ in range(len(test_data))]))
            #     imputed_data = np.stack(imputed_data, axis=-1)
            elif imputation_method == "emd":
                imputed_data = []
                for train_col in train_data.T:
                    imputed_data.append(np.random.choice(train_col, len(test_data)))
                imputed_data = np.stack(imputed_data, axis=-1)
        return imputed_data


    @staticmethod
    def split_dataset_by_natural_shift(x, y, cat_indices, shift_type, shift_severity, regression):
        from xgboost import XGBClassifier, XGBRegressor
        from sklearn.preprocessing import LabelEncoder

        # preprocess input and output
        for cat_index in cat_indices:
            x.iloc[:, cat_index] = x.iloc[:, cat_index].astype('category').cat.codes
        x_train = x.to_numpy()
        le = LabelEncoder()
        y_train = le.fit_transform(y.to_numpy())

        # fit on xgboost
        xgb = XGBClassifier() if not regression else XGBRegressor()
        xgb.fit(x_train, y_train)

        if shift_type == "numerical":
            cont_indices = np.array(sorted(set(np.arange(x.shape[-1])).difference(set(cat_indices))))
            important_feature_idx = cont_indices[np.argmax(xgb.feature_importances_[cont_indices])]
            x = x.sort_values(by=[x.columns[important_feature_idx]], ascending=False)
        elif shift_type == "categorical":
            important_feature_idx = cat_indices[np.argmax(xgb.feature_importances_[cat_indices])]
            x[f"{x.columns[important_feature_idx]}_count"] = x.groupby(x.columns[important_feature_idx])[x.columns[important_feature_idx]].transform('count')
            x = x.sort_values(by=[f"{x.columns[important_feature_idx]}_count"], ascending=False)
        else:
            raise NotImplementedError

        train_indices = np.concatenate([
            np.random.choice(x.iloc[:int(x.shape[0] * 0.8), :].index, size=int(x.shape[0] * 0.8 * (0.8 + 0.2 * shift_severity)), replace=False),
            np.random.choice(x.iloc[int(x.shape[0] * 0.8):, :].index, size=int(x.shape[0] * 0.8 * 0.2 * (1 - shift_severity)), replace=False),
        ])
        test_indices = np.array(sorted(set(np.arange(x.shape[0])).difference(set(train_indices))))
        return train_indices, test_indices


    @staticmethod
    def revert_recon_to_onehot(reconstructed_data, cat_indices_groups):
        if len(reconstructed_data.shape) == 1:
            reconstructed_data = reconstructed_data.unsqueeze(0)

        new_data = reconstructed_data.clone()
        for group in cat_indices_groups:
            cat_part = reconstructed_data[:, group]
            cat_part_softmax = F.softmax(cat_part, dim=1)
            new_data[:, group] = cat_part_softmax
        new_data = new_data.squeeze()
        return new_data