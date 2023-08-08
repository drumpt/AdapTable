import os
from os import path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tableshift"))
from collections import Counter

import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch

import openml
from openml import tasks, runs
from data.utils.util_functions import load_opt
from datasets import load_dataset
from tableshift import get_dataset, get_iid_dataset
from utils import utils



class Dataset():
    def __init__(self, args):
        # if args.seed:
        #     utils.set_seed(args.seed)
        if args.benchmark == "openml-cc18":
            self.dataset = OpenMLCC18Dataset(args)
        elif args.benchmark == "tableshift":
            self.dataset = TableShiftDataset(args)
        elif args.benchmark == "openml-regression":
            self.dataset = OpenMLRegressionDataset(args)
        elif args.benchmark == "shifts":
            self.dataset = ShiftsDataset(args)
        elif args.benchmark == "folktables":
            self.dataset = FolkTablesDataset(args)
        elif args.benchmark == "scikit-learn":
            self.dataset = ScikitLearnDataset(args)
        else:
            raise NotImplementedError

        if float(args.train_ratio) != 1:
            len_of_train = len(self.dataset.train_x)
            self.dataset.train_x = self.dataset.train_x[:int(len_of_train * float(args.train_ratio))]
            self.dataset.train_y = self.dataset.train_y[:int(len_of_train * float(args.train_ratio))]

        train_data = torch.utils.data.TensorDataset(torch.FloatTensor(self.dataset.train_x), torch.FloatTensor(self.dataset.train_y))
        valid_data = torch.utils.data.TensorDataset(torch.FloatTensor(self.dataset.valid_x), torch.FloatTensor(self.dataset.valid_y))
        test_data = torch.utils.data.TensorDataset(torch.FloatTensor(self.dataset.test_x), torch.FloatTensor(self.dataset.test_mask_x), torch.FloatTensor(self.dataset.test_y))

        print(f"dataset size - train: {len(self.dataset.train_x)}, valid: {len(self.dataset.valid_x)}, test: {len(self.dataset.test_x)}")

        self.in_dim, self.out_dim = self.dataset.train_x.shape[-1], self.dataset.train_y.shape[-1]
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, worker_init_fn=utils.set_seed_worker, generator=utils.get_generator(args.seed))
        self.valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.train_batch_size, shuffle=True, worker_init_fn=utils.set_seed_worker, generator=utils.get_generator(args.seed))
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True, worker_init_fn=utils.set_seed_worker, generator=utils.get_generator(args.seed))



class ScikitLearnDataset():
    def __init__(self, args):
        from sklearn import datasets
        if args.dataset == 'make_classification':
            # n_features = number of independent features
            # n_redundant = number of redundant features
            # n_informative = number of informative features
            # class_sep = default 1, where larger value makes classification easier
            n_features = 30
            n_informative = 5
            class_sep = 1
            n_redundant = n_features - n_informative
            x,y = sklearn.datasets.make_classification(n_samples=5000, class_sep=class_sep, n_classes=10, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, random_state=args.seed, shuffle=True)
        elif args.dataset == 'two_moons':
            # noise = amount of noise added to moons dataset
            x,y = sklearn.datasets.make_moons(n_samples=5000, random_state=args.seed, noise=0.3, shuffle=True)
        else:
            raise NotImplementedError

        y = y.reshape(-1, 1)
        self.visualize_dataset(x, y)
        # train/valid/test split
        train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4, random_state=args.seed)
        valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y, test_size=0.5, random_state=args.seed)


        self.input_scaler = getattr(sklearn.preprocessing, args.normalizer)()
        self.input_scaler.fit(
            np.concatenate([train_x, valid_x], axis=0)
        )
        train_x = self.input_scaler.transform(train_x)
        valid_x = self.input_scaler.transform(valid_x)
        test_x, test_mask_x = get_corrupted_data(np.array(test_x),
                                                           np.array(train_x),
                                                           data_type="numerical", shift_type=args.shift_type,
                                                           shift_severity=args.shift_severity,
                                                           imputation_method=args.imputation_method)
        test_x = self.input_scaler.transform(test_x)

        self.train_x = np.concatenate([train_x], axis=-1)
        self.valid_x = np.concatenate([valid_x], axis=-1)
        self.test_x = np.concatenate([test_x], axis=-1)
        self.test_mask_x =  np.concatenate([test_x], axis=-1)

        self.output_one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.output_one_hot_encoder.fit(np.concatenate([train_y, valid_y], axis=0))
        self.train_y = self.output_one_hot_encoder.transform(train_y)
        self.valid_y = self.output_one_hot_encoder.transform(valid_y)
        self.test_y = self.output_one_hot_encoder.transform(test_y)

    def visualize_dataset(self, X, y):
        import matplotlib.pyplot as plt
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=100,
                    edgecolor="k", linewidth=2)
        plt.xlabel("$X_1$")
        plt.ylabel("$X_2$")
        plt.show()



class OpenMLCC18Dataset():
    def __init__(self, args):
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
        print(f"x: {x}")
        y = np.array(y).reshape(-1, 1)

        # train/valid/test split
        train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4, random_state=42)
        valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y, test_size=0.5, random_state=42)

        # data preprocessing (normalization and one-hot encoding)
        cat_indices = np.argwhere(np.array(cat_indicator) == True).flatten()
        cont_indices = np.array(sorted(set(np.arange(x.shape[1])).difference(set(cat_indices))))
        if len(cont_indices):
            self.input_scaler = getattr(sklearn.preprocessing, args.normalizer)()
            self.input_scaler.fit(
                np.concatenate([train_x.iloc[:, cont_indices], valid_x.iloc[:, cont_indices]], axis=0)
            )
            train_cont_x = self.input_scaler.transform(train_x.iloc[:, cont_indices])
            valid_cont_x = self.input_scaler.transform(valid_x.iloc[:, cont_indices])
            test_cont_x, test_cont_mask_x = get_corrupted_data(np.array(test_x.iloc[:, cont_indices]), np.array(train_x.iloc[:, cont_indices]), data_type="numerical", shift_type=args.shift_type, shift_severity=args.shift_severity, imputation_method=args.imputation_method)
            test_cont_x = self.input_scaler.transform(test_cont_x)
        else:
            train_cont_x, valid_cont_x, test_cont_x, test_cont_mask_x = np.array([]), np.array([]), np.array([]), np.array([])
        if len(cat_indices):
            self.input_one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.input_one_hot_encoder.fit(
                np.concatenate([train_x.iloc[:, cat_indices], valid_x.iloc[:, cat_indices]], axis=0)
            )
            train_cat_x = self.input_one_hot_encoder.transform(train_x.iloc[:, cat_indices])
            valid_cat_x = self.input_one_hot_encoder.transform(valid_x.iloc[:, cat_indices])
            test_cat_x, test_cat_mask_x = get_corrupted_data(np.array(test_x.iloc[:, cat_indices]), np.array(train_x.iloc[:, cat_indices]), data_type="categorical", shift_type=args.shift_type, shift_severity=args.shift_severity, imputation_method=args.imputation_method)
            test_cat_x = self.input_one_hot_encoder.transform(test_cat_x)
            test_cat_mask_x = np.concatenate([np.repeat(test_cat_mask_x[:, category_idx][:, None], len(category), axis=1) for category_idx, category in enumerate(self.input_one_hot_encoder.categories_)], axis=1)
        else:
            train_cat_x, valid_cat_x, test_cat_x, test_cat_mask_x = np.array([]), np.array([]), np.array([]), np.array([])

        self.train_x = np.concatenate([
            train_cont_x if len(cont_indices) else train_cont_x.reshape(train_cat_x.shape[0], 0),
            train_cat_x if len(cat_indices) else train_cat_x.reshape(train_cont_x.shape[0], 0)
        ], axis=-1)
        self.valid_x = np.concatenate([
            valid_cont_x if len(cont_indices) else valid_cont_x.reshape(valid_cat_x.shape[0], 0),
            valid_cat_x if len(cat_indices) else valid_cat_x.reshape(valid_cont_x.shape[0], 0)
        ], axis=-1)
        self.test_x = np.concatenate([
            test_cont_x if len(cont_indices) else test_cont_x.reshape(test_cat_x.shape[0], 0),
            test_cat_x if len(cat_indices) else test_cat_x.reshape(test_cont_x.shape[0], 0)
        ], axis=-1)
        self.test_mask_x = np.concatenate([
            test_cont_mask_x if len(cont_indices) else test_cont_mask_x.reshape(test_cat_mask_x.shape[0], 0),
            test_cat_mask_x if len(cat_indices) else test_cat_mask_x.reshape(test_cont_mask_x.shape[0], 0)
        ], axis=-1)

        self.output_one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.output_one_hot_encoder.fit(np.concatenate([train_y, valid_y], axis=0))
        self.train_y = self.output_one_hot_encoder.transform(train_y)
        self.valid_y = self.output_one_hot_encoder.transform(valid_y)
        self.test_y = self.output_one_hot_encoder.transform(test_y)



class TableShiftDataset():
    def __init__(self, args):
        dataset = get_dataset(
            args.dataset,
            cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tableshift/tableshift/tmp"),
        )
        if dataset.is_domain_split:
            print(f'ood dataset')
        else:
            print(f'not ood')

        train_x, train_y, _, _ = dataset.get_pandas("train")
        valid_x, valid_y, _, _ = dataset.get_pandas("validation")

        # TODO: require kaggle.json file!
        if dataset.is_domain_split:
            test_x, test_y, _, _ = dataset.get_pandas("ood_test")
        else:
            test_x, test_y, _, _ = dataset.get_pandas("test")

        self.train_x = np.array(train_x[sorted(train_x.columns)])
        self.valid_x = np.array(valid_x[sorted(valid_x.columns)])
        self.test_x = np.array(test_x[sorted(test_x.columns)])
        self.test_mask_x = np.ones_like(self.test_x)

        train_y = np.array(train_y).reshape(-1, 1)
        valid_y = np.array(valid_y).reshape(-1, 1)
        test_y = np.array(test_y).reshape(-1, 1)

        self.output_one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.output_one_hot_encoder.fit(np.concatenate([train_y, valid_y], axis=0))
        self.train_y = self.output_one_hot_encoder.transform(train_y)
        self.valid_y = self.output_one_hot_encoder.transform(valid_y)
        self.test_y = self.output_one_hot_encoder.transform(test_y)



class FolkTablesDataset():
    def __init__(self, args):
        from folktables import ACSDataSource, ACSIncome, ACSPublicCoverage
        if args.dataset == 'state':
            data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
            ca_data = data_source.get_data(states=["CA"], download=True)
            mi_data = data_source.get_data(states=["MI"], download=True)
            train_x, train_y, _ = ACSIncome.df_to_numpy(ca_data)
            test_x, test_y, _ = ACSIncome.df_to_numpy(mi_data)
            train_y = np.int32(train_y).reshape(-1, 1)
            test_y = np.int32(test_y).reshape(-1, 1)
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

        elif args.dataset == 'time':
            # for training
            train_x_list = []
            train_y_list = []
            for year in [2014, 2015, 2016]:
                data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
                acs_data = data_source.get_data(states=["CA"], download=True)
                features, labels, _ = ACSPublicCoverage.df_to_numpy(acs_data)
                train_x_list.append(features)
                train_y_list.append(labels)

            test_x_list = []
            test_y_list = []
            # for testing
            for year in [2017, 2018]:
                data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
                acs_data = data_source.get_data(states=["CA"], download=True)
                features, labels, _ = ACSPublicCoverage.df_to_numpy(acs_data)
                test_x_list.append(features)
                test_y_list.append(labels)

            train_x = np.concatenate(train_x_list, axis=0)
            train_y = np.concatenate(train_y_list, axis=0)
            test_x = np.concatenate(test_x_list, axis=0)
            test_y = np.concatenate(test_y_list, axis=0)

            # for one-hot encoding
            train_y = np.int32(train_y).reshape(-1, 1)
            test_y = np.int32(test_y).reshape(-1, 1)

            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
            # raise NotImplementedError

        elif args.dataset == 'state_time' or args.dataset == 'time_state':
            # for training
            train_x_list = []
            train_y_list = []
            for year in [2014, 2015, 2016]:
                data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
                acs_data = data_source.get_data(states=["CA"], download=True)
                features, labels, _ = ACSPublicCoverage.df_to_numpy(acs_data)
                train_x_list.append(features)
                train_y_list.append(labels)

            test_x_list = []
            test_y_list = []
            # for testing
            for year in [2017, 2018]:
                data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
                acs_data = data_source.get_data(states=["MI"], download=True)
                features, labels, _ = ACSPublicCoverage.df_to_numpy(acs_data)
                test_x_list.append(features)
                test_y_list.append(labels)

            train_x = np.concatenate(train_x_list, axis=0)
            train_y = np.concatenate(train_y_list, axis=0)
            test_x = np.concatenate(test_x_list, axis=0)
            test_y = np.concatenate(test_y_list, axis=0)

            # for one-hot encoding
            train_y = np.int32(train_y).reshape(-1, 1)
            test_y = np.int32(test_y).reshape(-1, 1)

            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
            # raise NotImplementedError

        # normalize input
        self.input_scaler = getattr(sklearn.preprocessing, args.normalizer)()
        self.input_scaler.fit(np.concatenate([train_x, valid_x], axis=0))
        self.train_x = self.input_scaler.transform(train_x)
        self.valid_x = self.input_scaler.transform(valid_x)
        self.test_x = self.input_scaler.transform(test_x)
        self.test_mask_x = np.ones_like(self.test_x)

        # one-hot encode output
        self.output_one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.output_one_hot_encoder.fit(np.concatenate([train_y, valid_y], axis=0))
        self.train_y = self.output_one_hot_encoder.transform(train_y)
        self.valid_y = self.output_one_hot_encoder.transform(valid_y)
        self.test_y = self.output_one_hot_encoder.transform(test_y)



class OpenMLRegressionDataset():
    def __init__(self, args):
        # conda install -c huggingface -c conda-forge datasets
        # TODO: give arguments as data_split(type of dataset) and dataset_specification(dataset name)
        dataset_specification = args.dataset

        from scipy.io import arff
        if dataset_specification in ["cholestrol", "sarcos", "boston", "news"]:
            config = load_opt(dataset_specification)

            data = arff.loadarff(config['path'])
            df = pd.DataFrame(data[0])

            if dataset_specification not in ['sarcos', 'news']:
                # load as dataframe and convert datatypes to float
                str_df = df.select_dtypes([object])
                str_df = str_df.stack().str.decode('utf-8').unstack()
                str_df.replace(to_replace='?', value=np.nan, inplace=True)
                df[str_df.columns] = str_df.astype(np.float64)

            dataset_df = df
            wo_target = list(dataset_df.columns[:-1])
            target = list([dataset_df.columns[-1]])

            x = dataset_df[wo_target]
            y = dataset_df[target]

            # train/valid/test split
            train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4, random_state=42)
            valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y, test_size=0.5, random_state=42)

            # remove nan in training
            tot_train = pd.concat([train_x, train_y], axis=1, join="inner")
            tot_train = tot_train.dropna()
            train_x = tot_train[wo_target]
            train_y = tot_train[target]

            tot_valid = pd.concat([valid_x, valid_y], axis=1, join="inner")
            tot_valid = tot_valid.dropna()
            valid_x = tot_valid[wo_target]
            valid_y = tot_valid[target]

            # is it OK?
            tot_test = pd.concat([test_x, test_y], axis=1, join="inner")
            tot_test = tot_test.dropna()
            test_x = tot_test[wo_target]
            test_y = tot_test[target]
        elif dataset_specification in ["abalone", "seattlecrime6", "diamonds", "Brazilian_houses", "topo_2_1", "house_sales", "particulate-matter-ukair-2017", "analcatdata_supreme", "delays_zurich_transport", "Bike_Sharing_Demand", "nyc-taxi-green-dec-2016", "visualizing_soil", "SGEMM_GPU_kernel_performance"]:
            dataset = load_dataset("inria-soda/tabular-benchmark", data_files=f"reg_cat/{dataset_specification}.csv")
            config = load_opt(dataset_specification)
            # dataset = load_dataset("inria_soda/tabular-benchmark", data_files=f"{data_split}/{args.dataset}.csv")
            # last column is target
            column_names = dataset['train'].column_names
            wo_target = dataset['train'].column_names[:-1]
            target = [dataset['train'].column_names[-1]]

            pandas_dataset = dataset['train'].to_pandas()
            x = pandas_dataset[wo_target]
            y = pandas_dataset[target]

            # train/valid/test split
            train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4, random_state=42)
            valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y, test_size=0.5, random_state=42)
        else:
            raise NotImplementedError

        # data preprocessing (normalization and one-hot encoding)
        cat_indicator = [True if column_name in config['nominal_columns'] else False for column_name in x.columns]
        cat_indices = np.argwhere(np.array(cat_indicator) == True).flatten()
        cont_indices = np.array(sorted(set(np.arange(x.shape[1])).difference(set(cat_indices))))

        if len(cont_indices):
            self.input_scaler = getattr(sklearn.preprocessing, args.normalizer)()
            self.input_scaler.fit(
                np.concatenate([train_x.iloc[:, cont_indices], valid_x.iloc[:, cont_indices]], axis=0))
            train_cont_x = self.input_scaler.transform(train_x.iloc[:, cont_indices])
            valid_cont_x = self.input_scaler.transform(valid_x.iloc[:, cont_indices])

            test_cont_x, test_cont_mask_x = get_corrupted_data(np.array(test_x.iloc[:, cont_indices]),
                                                               np.array(train_x.iloc[:, cont_indices]),
                                                               data_type="numerical", shift_type=args.shift_type,
                                                               shift_severity=args.shift_severity,
                                                               imputation_method=args.imputation_method)
            test_cont_x = self.input_scaler.transform(test_cont_x)
        else:
            train_cont_x, test_cont_mask_x, valid_cont_x, test_cont_x = np.array([]), np.array([]), np.array([]), np.array([])
        if len(cat_indices):
            self.input_one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.input_one_hot_encoder.fit(
                np.concatenate([train_x.iloc[:, cat_indices], valid_x.iloc[:, cat_indices]], axis=0))
            train_cat_x = self.input_one_hot_encoder.transform(train_x.iloc[:, cat_indices])
            valid_cat_x = self.input_one_hot_encoder.transform(valid_x.iloc[:, cat_indices])
            test_cat_x, test_cat_mask_x = get_corrupted_data(np.array(test_x.iloc[:, cat_indices]),
                                                             np.array(train_x.iloc[:, cat_indices]),
                                                             data_type="categorical", shift_type=args.shift_type,
                                                             shift_severity=args.shift_severity,
                                                             imputation_method=args.imputation_method)
            test_cat_x = self.input_one_hot_encoder.transform(test_cat_x)
            test_cat_mask_x = np.concatenate([np.repeat(test_cat_mask_x[:, category_idx][:, None], len(category), axis=1) for category_idx, category in enumerate(self.input_one_hot_encoder.categories_)], axis=1)
        else:
            train_cat_x, valid_cat_x, test_cat_x, test_cat_mask_x = np.array([]), np.array([]), np.array([]), np.array([])

        self.train_x = np.concatenate([
            train_cont_x if len(cont_indices) else train_cont_x.reshape(train_cat_x.shape[0], 0),
            train_cat_x if len(cat_indices) else train_cat_x.reshape(train_cont_x.shape[0], 0)
        ], axis=-1)
        self.valid_x = np.concatenate([
            valid_cont_x if len(cont_indices) else valid_cont_x.reshape(valid_cat_x.shape[0], 0),
            valid_cat_x if len(cat_indices) else valid_cat_x.reshape(valid_cont_x.shape[0], 0)
        ], axis=-1)
        self.test_x = np.concatenate([
            test_cont_x if len(cont_indices) else test_cont_x.reshape(test_cat_x.shape[0], 0),
            test_cat_x if len(cat_indices) else test_cat_x.reshape(test_cont_x.shape[0], 0)
        ], axis=-1)
        self.test_mask_x = np.concatenate([
            test_cont_mask_x if len(cont_indices) else test_cont_mask_x.reshape(test_cat_mask_x.shape[0], 0),
            test_cat_mask_x if len(cat_indices) else test_cat_mask_x.reshape(test_cont_mask_x.shape[0], 0)
        ], axis=-1)

        self.output_scaler = getattr(sklearn.preprocessing, args.normalizer)()
        self.output_scaler.fit(np.concatenate([train_y, valid_y], axis=0))
        self.train_y = self.output_scaler.transform(train_y)
        self.valid_y = self.output_scaler.transform(valid_y)
        self.test_y = self.output_scaler.transform(test_y)



class ShiftsDataset():
    def __init__(self, args):
        if 'weather' in args.dataset:
            dataset_path_name = 'weather'
        else:
            dataset_path_name = args.dataset

        if (not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"shifts/{dataset_path_name}/train.csv"))) and 'weather' in args.dataset:
            csv_list = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"shifts/{dataset_path_name}/shifts_canonical_train.csv"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"shifts/{dataset_path_name}/shifts_canonical_dev_in.csv"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"shifts/{dataset_path_name}/shifts_canonical_eval_in.csv"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"shifts/{dataset_path_name}/shifts_canonical_eval_out.csv"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"shifts/{dataset_path_name}/shifts_canonical_dev_out.csv"),
            ]
            train_df, valid_df, test_df = self.concat_csvs(csv_list, args)
            train_df = train_df
            valid_df = valid_df
            test_df = test_df

        else:
            train_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                f"shifts/{dataset_path_name}/train.csv")).dropna()
            valid_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                f"shifts/{dataset_path_name}/dev_in.csv")).dropna()
            test_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                f"shifts/{dataset_path_name}/dev_out.csv")).dropna()

        if args.dataset == 'weather_reg':
            train_x = train_df[train_df.columns.difference(['fact_temperature', 'climate'])].astype(np.float64)
            valid_x = valid_df[train_df.columns.difference(['fact_temperature', 'climate'])].astype(np.float64)
            test_x = test_df[train_df.columns.difference(['fact_temperature', 'climate'])].astype(np.float64)
            train_y = train_df['fact_temperature'].astype(np.float64)
            valid_y = valid_df['fact_temperature'].astype(np.float64)
            test_y = test_df['fact_temperature'].astype(np.float64)
        elif args.dataset == 'weather_cls':
            train_x = train_df[train_df.columns.difference(['fact_cwsm_class', 'climate'])].astype(np.float64)
            valid_x = valid_df[train_df.columns.difference(['fact_cwsm_class', 'climate'])].astype(np.float64)
            test_x = test_df[train_df.columns.difference(['fact_cwsm_class', 'climate'])].astype(np.float64)
            train_y = train_df['fact_cwsm_class'].astype(np.float64)
            valid_y = valid_df['fact_cwsm_class'].astype(np.float64)
            test_y = test_df['fact_cwsm_class'].astype(np.float64)
        elif args.dataset == 'power':
            train_x = train_df[train_df.columns.difference(['power'])].astype(np.float64)
            valid_x = valid_df[train_df.columns.difference(['power'])].astype(np.float64)
            test_x = test_df[train_df.columns.difference(['power'])].astype(np.float64)
            train_y = train_df['power'].astype(np.float64)
            valid_y = valid_df['power'].astype(np.float64)
            test_y = test_df['power'].astype(np.float64)
        print('loading done!')

        self.input_scaler = getattr(sklearn.preprocessing, args.normalizer)()
        self.input_scaler.fit(np.concatenate([train_x, valid_x], axis=0))

        self.train_x = self.input_scaler.transform(train_x)
        self.valid_x = self.input_scaler.transform(valid_x)
        test_x = self.input_scaler.transform(test_x)
        self.test_x, self.test_mask_x = get_corrupted_data(np.array(test_x),
                                                      np.array(train_x),
                                                      data_type="numerical", shift_type=args.shift_type,
                                                      shift_severity=args.shift_severity,
                                                      imputation_method=args.imputation_method)

        if 'cls' in args.dataset:
            self.output_one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.output_one_hot_encoder.fit(np.concatenate([train_y, valid_y], axis=0).reshape(-1, 1))
            self.train_y = self.output_one_hot_encoder.transform(np.array(train_y).reshape(-1, 1))
            self.valid_y = self.output_one_hot_encoder.transform(np.array(valid_y).reshape(-1, 1))
            self.test_y = self.output_one_hot_encoder.transform(np.array(test_y).reshape(-1, 1))
        else:
            self.output_scaler = getattr(sklearn.preprocessing, args.normalizer)()
            self.output_scaler.fit(np.concatenate([train_y, valid_y], axis=0))
            self.train_y = self.output_scaler.transform(train_y).reshape(-1, 1)
            self.valid_y = self.output_scaler.transform(valid_y).reshape(-1, 1)
            self.test_y = self.output_scaler.transform(test_y).reshape(-1, 1)

        print('dataset preprocess done!')


    def concat_csvs(self, path_list, args):
        import pandas as pd
        pd_list = []
        for path in path_list:
            pd_list.append(pd.read_csv(path))
        train_pd = pd.concat(pd_list[:3])
        train_pd.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                f"shifts/weather/train.csv"))
        eval_pd = pd.concat([pd_list[3]])
        eval_pd.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                f"shifts/weather/dev_in.csv"))
        test_pd = pd.concat(pd_list[4:])
        test_pd.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                f"shifts/weather/dev_out.csv"))
        return (train_pd, eval_pd, test_pd)


def get_corrupted_data(test_data, train_data, data_type, shift_type, shift_severity, imputation_method):
    mask = None

    if shift_type == "Gaussian" and data_type == "numerical":
        scaler = StandardScaler()
        scaler.fit(train_data)
        test_data = test_data.astype(np.float64) + shift_severity * np.random.randn(*test_data.shape) * np.sqrt(scaler.var_)

    elif shift_type in ["random_drop", "column_drop", "random_replacement", "column_replacement"]:
        assert 0 <= shift_severity <= 1

        if shift_type == "random_drop":
            mask = (np.random.rand(*test_data.shape) >= shift_severity).astype(np.int64)
        elif shift_type == "column_drop":
            mask = np.repeat((np.random.rand(*test_data.shape[1:]) >= shift_severity).astype(np.int64)[None, :], test_data.shape[0], axis=0)
        else:
            mask = np.ones_like(test_data, dtype=np.int64)            
        imputed_data = get_imputed_data(test_data, train_data, data_type, imputation_method)

        if not isinstance(test_data, np.ndarray):
            test_data = test_data.detach().cpu().numpy()

        if data_type == "numerical":
            test_data = mask * test_data + (1 - mask) * imputed_data
        elif data_type == "categorical":
            for row_idx in range(test_data.shape[0]):
                for col_idx in range(test_data.shape[-1]):
                    if mask[row_idx][col_idx] == 0:
                        test_data[row_idx][col_idx] = imputed_data[row_idx][col_idx]

    elif shift_type in ["mean_shift", "std_shift", "mean_std_shift"] and data_type == "numerical":
        assert 0 <= shift_severity <= 1
        mask = np.ones_like(test_data, dtype=np.int64)
        # mask[:, int(np.random.choice(test_data.shape[1], 1))] = 0 # select only one column
        mask = np.repeat((np.random.rand(*test_data.shape[1:]) >= shift_severity).astype(np.int64)[None, :], test_data.shape[0], axis=0)
        mean_severity_const = 0.01
        std_severity_const = 0.001

        scaler = StandardScaler()
        scaler.fit(train_data)
        if shift_type == "mean_shift":
            mean_noise = mean_severity_const * np.random.randn(*scaler.var_.shape)
            test_data = mask * test_data + (1 - mask) * (test_data + mean_noise * np.sqrt(scaler.var_))
        elif shift_type == "std_shift":
            std_noise = np.exp(std_severity_const * np.random.randn(*scaler.var_.shape))
            test_data = mask * test_data + (1 - mask) * (std_noise * test_data + scaler.mean_ * (1 - std_noise))
        elif shift_type == "mean_std_shift":
            mean_noise = mean_severity_const * np.random.randn(*scaler.mean_.shape)
            std_noise = np.exp(std_severity_const * np.random.randn(*scaler.var_.shape))
            test_data = mask * test_data + (1 - mask) * (std_noise * test_data + mean_noise * np.sqrt(scaler.var_) + (1 - std_noise) * scaler.mean_)
        mask = np.ones_like(test_data, dtype=np.int64)

    if mask is None:
        mask = np.ones_like(test_data, dtype=np.int64)
    return test_data, mask


def get_imputed_data(test_data, train_data, data_type, imputation_method):
    if data_type == "numerical":
        if imputation_method == "zero":
            if isinstance(test_data, torch.Tensor):
                test_data = test_data.cpu()
            imputed_data = np.zeros_like(test_data)
        elif imputation_method == "mean":
            imputed_data = np.repeat(np.mean(train_data, axis=0)[None, :], len(test_data), axis=0)
        elif imputation_method == "emd":
            imputed_data = []
            if isinstance(test_data, torch.Tensor):
                test_data = test_data.cpu()
            # for test_col in test_data.T:
            #     imputed_data.append(np.random.choice(test_col, len(test_data)))
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
        elif imputation_method == "mean":  # mode (most frequent value)
            imputed_data = []
            for train_col in train_data.T:
                unique, counts = list(Counter(train_col).keys()), list(Counter(train_col).values())
                imputed_data.append(np.array([unique[np.argmax(counts)] for _ in range(len(test_data))]))
            imputed_data = np.stack(imputed_data, axis=-1)
        elif imputation_method == "emd":
            imputed_data = []
            for train_col in train_data.T:
                imputed_data.append(np.random.choice(train_col, len(test_data)))
            imputed_data = np.stack(imputed_data, axis=-1)
    return imputed_data


def get_mask_by_feature_importance(args, test_data, importance):
    mask = torch.ones_like(test_data, dtype=torch.float32)
    selected_rows = np.random.choice(test_data.shape[0], size=int(len(test_data.flatten()) * args.mask_ratio))
    selected_columns = np.random.choice(test_data.shape[-1], size=int(len(test_data.flatten()) * args.mask_ratio), p=importance.cpu().numpy())
    mask[selected_rows, selected_columns] = 0
    return mask