import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tableshift"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tableshift/tableshift"))
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import norm, multinomial, dirichlet
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
import torch.distributions as dist
import openml
from openml import tasks, runs

from tableshift import get_dataset, get_iid_dataset
from data.utils.util_functions import load_opt
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
            train_x = train_x[:int(len(train_x) * float(args.train_ratio)), :]
            train_y = train_y[:int(len(train_y) * float(args.train_ratio)), :]

        if args.smote:
            from imblearn.over_sampling import SMOTENC, SMOTE
            if len(cat_indices):
                train_x, train_y = SMOTENC(categorical_features=cat_indices, random_state=args.seed).fit_resample(train_x, train_y)
            else:
                train_x, train_y = SMOTE(random_state=args.seed).fit_resample(train_x, train_y)

        train_x, train_y = pd.DataFrame(train_x), pd.DataFrame(train_y)
        valid_x, valid_y = pd.DataFrame(valid_x), pd.DataFrame(valid_y)
        test_x, test_y = pd.DataFrame(test_x), pd.DataFrame(test_y)

        ##### preprocessing #####
        cont_indices = np.array(sorted(set(np.arange(train_x.shape[-1])).difference(set(cat_indices))))
        train_x.iloc[:, cont_indices] = train_x.iloc[:, cont_indices].fillna(0).astype(float)
        valid_x.iloc[:, cont_indices] = valid_x.iloc[:, cont_indices].fillna(0).astype(float)
        test_x.iloc[:, cont_indices] = test_x.iloc[:, cont_indices].fillna(0).astype(float)
        # train_y, valid_y, test_y = train_y.astype(float), valid_y.astype(float), test_y.astype(float)
        test_sampler = self.get_sampler(args, train_x, train_y, valid_x, valid_y, test_x, test_y, cat_indices)

        self.emb_dim = []
        if len(cont_indices):
            self.input_scaler = getattr(sklearn.preprocessing, args.normalizer)()
            self.input_scaler.fit(np.concatenate([train_x.iloc[:, cont_indices], valid_x.iloc[:, cont_indices]], axis=0))
            train_cont_x = self.input_scaler.transform(train_x.iloc[:, cont_indices])
            valid_cont_x = self.input_scaler.transform(valid_x.iloc[:, cont_indices])
            # if not args.benchmark in ["openml-cc18", "openml-regression", "scikit-learn"]: # important: add new synthetic corruption benchmarks
            #     args.shift_type = None
            test_cont_x, test_cont_mask_x = Dataset.get_corrupted_data_by_modality(
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
            # if not args.benchmark in ["openml-cc18", "openml-regression", "scikit-learn"]: # important: add new synthetic corruption benchmarks
            #     args.shift_type = None
            test_cat_x, test_cat_mask_x = Dataset.get_corrupted_data_by_modality(
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

        self.regression = regression
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
        self.train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.train_batch_size,
            shuffle=True,
            worker_init_fn=utils.set_seed_worker,
            generator=utils.get_generator(args.seed)
        )
        self.valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=args.train_batch_size,
            shuffle=False,
            worker_init_fn=utils.set_seed_worker,
            generator=utils.get_generator(args.seed)
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=args.test_batch_size,
            shuffle=False,
            worker_init_fn=utils.set_seed_worker,
            generator=utils.get_generator(args.seed),
            sampler=test_sampler
        )

        self.cont_dim = train_cont_x.shape[-1]
        if hasattr(self, 'input_one_hot_encoder'):
            self.emb_dim_list = [(len(category), min(10, (len(category) + 1) // 2)) if len(cat_indices) else (0, 0) for category in self.input_one_hot_encoder.categories_]
            self.cat_end_indices = np.cumsum([num_category for num_category, _ in self.emb_dim_list])
            self.cat_start_indices = np.concatenate([[0], self.cat_end_indices], axis=0)[:-1]
            self.cat_indices_groups = [list(range(self.cont_dim + cat_start_index, self.cont_dim + cat_end_index)) for cat_start_index, cat_end_index in zip(self.cat_start_indices, self.cat_end_indices)]
        else:
            self.emb_dim_list, self.cat_end_indices, self.cat_start_indices, self.cat_indices_groups = [], np.array([]), np.array([]), []
        self.shift_at = -1

        # print dataset info
        self.train_counts = np.unique(np.argmax(self.train_y, axis=1), return_counts=True)
        self.valid_counts = np.unique(np.argmax(self.valid_y, axis=1), return_counts=True)
        self.test_counts = np.unique(np.argmax(self.test_y, axis=1), return_counts=True)

        self.posttrain_loader = torch.utils.data.DataLoader(train_data,  batch_size=args.test_batch_size, shuffle=True, worker_init_fn=utils.set_seed_worker, generator=utils.get_generator(args.seed))
        self.posttrain_valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.test_batch_size, shuffle=False, worker_init_fn=utils.set_seed_worker, generator=utils.get_generator(args.seed))

        logger.info(f"dataset size | train: {len(self.train_x)}, valid: {len(self.valid_x)}, test: {len(self.test_x)}")
        logger.info(f"Class distribution - train {np.round(self.train_counts[1] / np.sum(self.train_counts[1]), 2)}, {self.train_counts}")
        logger.info(f"Class distribution - valid {np.round(self.valid_counts[1] / np.sum(self.valid_counts[1]), 2)}, {self.valid_counts}")
        logger.info(f"Class distribution - test {np.round(self.test_counts[1] / np.sum(self.test_counts[1]), 2)}, {self.test_counts}")


    def get_sampler(self, args, train_x, train_y, valid_x, valid_y, test_x, test_y, cat_indices):
        train_x, train_y, valid_x, valid_y, test_x, test_y = np.array(train_x), np.array(train_y).squeeze(), np.array(valid_x), np.array(valid_y).squeeze(), np.array(test_x), np.array(test_y).squeeze()
        if args.shift_type == "temp_corr":
            sampler = TemporallyCorrelatedSampler(test_y, args.temp_corr_alpha, args.temp_corr_window_size)
        elif args.shift_type == "imbalanced":
            sampler = ImbalancedSampler(train_y, args.imb_ratio)
        elif args.shift_type == "numerical" or args.shift_type == "categorical":
            from xgboost import XGBClassifier

            for cat_index in cat_indices:
                input_le = LabelEncoder()
                input_le.fit(np.concatenate([
                    train_x[:, cat_index], valid_x[:, cat_index], test_x[:, cat_index]
                ]))
                train_x[:, cat_index] = input_le.transform(train_x[:, cat_index]).astype(np.int32)
                valid_x[:, cat_index] = input_le.transform(valid_x[:, cat_index]).astype(np.int32)
                test_x[:, cat_index] = input_le.transform(test_x[:, cat_index]).astype(np.int32)
            train_x = train_x.astype(np.float32)
            valid_x = valid_x.astype(np.float32)
            test_x = test_x.astype(np.float32)

            output_le = LabelEncoder()
            output_le.fit(np.concatenate([train_y, valid_y, test_y]))
            train_y = output_le.transform(train_y)
            valid_y = output_le.transform(valid_y)
            test_y = output_le.transform(test_y)
            xgb = XGBClassifier()
            xgb.fit(train_x, train_y)

            if args.shift_type == "numerical":
                cont_indices = np.array(sorted(set(np.arange(train_x.shape[-1])).difference(set(cat_indices))))
                important_feature_idx = cont_indices[np.argmax(xgb.feature_importances_[cont_indices])]

                mean = np.mean(train_x[:, important_feature_idx])
                std = np.std(train_x[:, important_feature_idx])

                likelihoods = []
                for numerical in test_x[:, important_feature_idx]:
                    likelihood = norm.pdf(
                        numerical,
                        mean,
                        std,
                    )
                    likelihoods.append(likelihood)
                likelihoods_numerical = np.array(likelihoods)
                sampler = InverseLikelihoodSampler(likelihoods_numerical)
            elif args.shift_type == "categorical":
                important_feature_idx = cat_indices[np.argmax(xgb.feature_importances_[cat_indices])]
                train_cat_encoded = train_x[:, important_feature_idx].astype(np.int32)
                test_cat_encoded = test_x[:, important_feature_idx].astype(np.int32)
                
                category_counts = np.bincount(train_cat_encoded, minlength=len(np.unique(train_cat_encoded))) + 1
                category_probs = category_counts / np.sum(category_counts)

                print(f"{train_cat_encoded=}")
                print(f"{category_counts=}")
                print(f"{category_probs=}")

                likelihoods = []
                for category in test_cat_encoded:
                    category_one_hot = np.zeros(len(category_counts))
                    category_one_hot[category] = 1
                    likelihood = multinomial.pmf(category_one_hot, n=np.sum(category_one_hot), p=category_probs)
                    likelihoods.append(likelihood)
                likelihoods_categorical = np.array(likelihoods)
                sampler = InverseLikelihoodSampler(likelihoods_categorical)
        else:
            sampler = UniformSampler(test_y)
        return sampler

    def get_shifted_column(self):
        if self.shift_at == -1:
            from utils.shift_severity import calculate_columnwise_kl_divergence
            kl_div_per_column = calculate_columnwise_kl_divergence(self.train_x, self.test_x)
            self.shift_at = np.argmax(kl_div_per_column)
        return self.shift_at


    def get_openml_cc18_dataset(self, args):
        def preprocess_dna_dataset(x):
            def binary_to_string(array):
                array = list(array)
                if array == ['1', '0', '0']:
                    return 0 # "A"
                elif array == ['0', '1', '0']:
                    return 1 # "T"
                elif array == ['0', '0', '1']:
                    return 2 # "G"
                else:
                    return 3 # "C"
            x = x.to_numpy()
            x_string = []
            for col_idx in range(0, x.shape[-1], 3):
                x_string.append(list(map(lambda x: binary_to_string(x), list(x[:, col_idx:col_idx + 3]))))
            x_string = pd.DataFrame(x_string).T
            cat_indicator = [True for _ in range(x_string.shape[-1])]
            return x_string, cat_indicator

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
        if args.dataset == "dna":
            x, cat_indicator = preprocess_dna_dataset(x)
        y = pd.DataFrame(y)
        cat_indices = np.argwhere(np.array(cat_indicator) == True).flatten()
        regression = False

        print(f"cat_indices: {cat_indices}")

        # if args.shift_type in ["numerical", "categorical"]:
        #     if (len(cat_indices) == x.shape[-1] and args.shift_type == "numerical") or (len(cat_indices) == 0 and args.shift_type == "categorical"):
        #         raise Exception(f'No {args.shift_type} columns in {args.dataset} dataset!')

        #     train_indices, test_indices = self.split_dataset_by_natural_shift(x, y, cat_indices, args.shift_type, args.shift_severity, regression=False)
        #     train_x, train_y = x.iloc[train_indices, :], y.iloc[train_indices, :]
        #     test_x, test_y = x.iloc[test_indices, :], y.iloc[test_indices, :]
        #     train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.25, random_state=42)
        # else:
        #     train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4, random_state=42)
        #     valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y, test_size=0.5, random_state=42)
        train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4, random_state=42)
        valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y, test_size=0.5, random_state=42)
        # print(f"train_x original: {train_x}")
        return (train_x, valid_x, test_x), (train_y, valid_y, test_y), cat_indices, regression

    def get_tableshift_dataset(self, args):
        # if args.shift_type in ["Gaussian", "uniform", "random_drop", "column_drop"]:
        #     dataset_dir = os.path.join(args.dataset_save_dir, args.benchmark, f"{args.dataset}_id_test.pkl")
        # else:
        #     dataset_dir = os.path.join(args.dataset_save_dir, args.benchmark, f"{args.dataset}.pkl")
        dataset_dir = os.path.join(args.dataset_save_dir, args.benchmark, f"{args.dataset}.pkl")

        if os.path.exists(dataset_dir):
            dataset_dict = pickle.load(open(dataset_dir, "rb"))
            train_x = dataset_dict["train_x"]
            valid_x = dataset_dict["valid_x"]
            test_x = dataset_dict["test_x"]
            train_y = dataset_dict["train_y"]
            valid_y = dataset_dict["valid_y"]
            test_y = dataset_dict["test_y"]
            cat_indices = dataset_dict["cat_indices"]
        else:
            # customize preprocessor
            from tableshift.core.features import get_categorical_columns
            from tableshift.configs.benchmark_configs import BENCHMARK_CONFIGS
            from tableshift.configs.non_benchmark_configs import NON_BENCHMARK_CONFIGS
            EXPERIMENT_CONFIGS = {**BENCHMARK_CONFIGS, **NON_BENCHMARK_CONFIGS}
            expt_config = EXPERIMENT_CONFIGS[args.dataset]
            preprocessor_config = expt_config.preprocessor_config
            preprocessor_config.categorical_features="passthrough"
            preprocessor_config.numeric_features="passthrough"

            dataset = get_dataset(args.dataset, cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tableshift/tableshift/tmp"), preprocessor_config=preprocessor_config)
            train_x, train_y, _, _ = dataset.get_pandas("train")
            valid_x, valid_y, _, _ = dataset.get_pandas("validation")

            # id여야 하는 경우: Gaussian, uniform, random_drop, column_drop
            # ood여야 하는 경우: null, numerical, categorical, temp corr, imbalanced
            # if args.shift_type in ["Gaussian", "uniform", "random_drop", "column_drop"]:
            #     test_x, test_y, _, _ = dataset.get_pandas("id_test")
            # else:
            #     test_x, test_y, _, _ = dataset.get_pandas("ood_test") if dataset.is_domain_split else dataset.get_pandas("id_test")
            test_x, test_y, _, _ = dataset.get_pandas("ood_test") if dataset.is_domain_split else dataset.get_pandas("id_test")

            cat_indices = np.array(sorted([train_x.columns.get_loc(c) for c in get_categorical_columns(train_x)]))

            train_x, valid_x, test_x = np.array(train_x), np.array(valid_x), np.array(test_x)
            train_y, valid_y, test_y = np.array(train_y)[:, None], np.array(valid_y)[:, None], np.array(test_y)[:, None]

            dataset_dict = dict()
            dataset_dict["train_x"] = train_x
            dataset_dict["valid_x"] = valid_x
            dataset_dict["test_x"] = test_x
            dataset_dict["train_y"] = train_y
            dataset_dict["valid_y"] = valid_y
            dataset_dict["test_y"] = test_y
            dataset_dict["cat_indices"] = cat_indices
            with open(dataset_dir, "wb") as f:
                pickle.dump(dataset_dict, f)

        print(f"{train_x.shape=}")
        print(f"{valid_x.shape=}")
        print(f"{test_x.shape=}")
        return (train_x, valid_x, test_x), (train_y, valid_y, test_y), cat_indices, False


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


    def get_corrupted_data(self, x, train_x, shift_type, shift_severity, imputation_method):
        if self.cont_dim and len(self.cat_indices_groups):
            cor_cont_x, cont_mask = Dataset.get_corrupted_data_by_modality(x[:, :self.cont_dim], train_x[:, :self.cont_dim], data_type="numerical", shift_type=shift_type, shift_severity=shift_severity, imputation_method=imputation_method)

            cor_cat_x, cat_mask = Dataset.get_corrupted_data_by_modality(self.input_one_hot_encoder.inverse_transform(x[:, self.cont_dim:].detach().cpu()), self.input_one_hot_encoder.inverse_transform(train_x[:, self.cont_dim:]), data_type="categorical", shift_type=shift_type, shift_severity=shift_severity, imputation_method=imputation_method)
            cor_cat_x = self.input_one_hot_encoder.transform(cor_cat_x)
            cat_mask = np.concatenate([np.repeat(cat_mask[:, category_idx][:, None], len(category), axis=1) for category_idx, category in enumerate(self.input_one_hot_encoder.categories_)], axis=1)

            cor_x = torch.FloatTensor(np.concatenate([cor_cont_x, cor_cat_x], axis=-1)).to(x.device)
            mask_x = torch.FloatTensor(np.concatenate([cont_mask, cat_mask], axis=-1)).to(x.device)
        elif self.cont_dim:
            cor_cont_x, cont_mask = Dataset.get_corrupted_data_by_modality(x[:, :self.cont_dim], train_x[:, :self.cont_dim], data_type="numerical", shift_type=shift_type, shift_severity=shift_severity, imputation_method=imputation_method)

            cor_x = torch.FloatTensor(cor_cont_x).to(x.device)
            mask_x = torch.FloatTensor(cont_mask).to(x.device)
        else:
            cor_cat_x, cat_mask = Dataset.get_corrupted_data_by_modality(self.input_one_hot_encoder.inverse_transform(x[:, self.cont_dim:].cpu()), self.input_one_hot_encoder.inverse_transform(train_x[:, self.cont_dim:]), data_type="categorical", shift_type=shift_type, shift_severity=shift_severity, imputation_method=imputation_method)
            cor_cat_x = self.input_one_hot_encoder.transform(cor_cat_x)
            cat_mask = np.concatenate([np.repeat(cat_mask[:, category_idx][:, None], len(category), axis=1) for category_idx, category in enumerate(self.input_one_hot_encoder.categories_)], axis=1)

            cor_x = torch.FloatTensor(cor_cat_x).to(x.device)
            mask_x = torch.FloatTensor(cat_mask).to(x.device)
        return cor_x, mask_x


    @staticmethod
    def get_corrupted_data_by_modality(test_data, train_data, data_type, shift_type, shift_severity, imputation_method):
        if torch.is_tensor(test_data):
            test_data = test_data.detach().cpu().numpy()
        mask = np.ones(test_data.shape, dtype=np.int32)
        if shift_type in ["Gaussian", "uniform"] and data_type == "numerical":
            scaler = StandardScaler()
            scaler.fit(train_data)
            noise = np.random.randn(*test_data.shape) if shift_type == "Gaussian" else np.random.uniform(low=-1, high=1, size=test_data.shape)
            test_data = test_data.astype(np.float32) + shift_severity * noise * np.sqrt(scaler.var_)
        elif shift_type in ["random_drop", "column_drop"]:
            assert 0 <= shift_severity <= 1
            if shift_type == "random_drop":
                len_keep = int(test_data.shape[-1] * (1 - shift_severity))
                idx = np.random.randn(*test_data.shape).argsort(axis=1)
                mask = np.take_along_axis(np.concatenate([np.ones((test_data.shape[0], len_keep)), np.zeros((test_data.shape[0], test_data.shape[-1] - len_keep))], axis=1), idx, axis=1)
            elif shift_type == "column_drop":
                len_keep = int(test_data.shape[-1] * (1 - shift_severity))
                mask = np.concatenate([np.ones(len_keep), np.zeros(test_data.shape[-1] - len_keep)])
                np.random.shuffle(mask)
                mask = np.repeat(mask[None, :], test_data.shape[0], axis=0)
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
                if isinstance(test_data, torch.Tensor):
                    test_data = test_data.cpu()
                imputed_data = np.zeros_like(test_data)
            elif imputation_method == "mean":
                imputed_data = np.repeat(np.mean(train_data, axis=0)[None, :], len(test_data), axis=0)
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
            elif imputation_method == "mean":  # mode (most frequent value) for categorical variable
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

    def renormalize_data(self, unscaled_data):
        if isinstance(unscaled_data, torch.Tensor):
            data = unscaled_data.clone().detach().cpu().numpy()
        elif isinstance(unscaled_data, np.ndarray):
            data = np.copy(unscaled_data)
        else:
            raise ValueError

        # cont_data = self.input_scaler.transform(data[:, :self.cont_dim])
        cont_data = data[:, :self.cont_dim]
        if hasattr(self, 'input_one_hot_encoder'):
            cat_data = self.input_one_hot_encoder.transform(data[:, self.cont_dim:])
        else:
            cat_data = []
        data = np.concatenate([cont_data, cat_data], axis=1)
        if isinstance(unscaled_data, torch.Tensor):
            data = torch.from_numpy(data).to(unscaled_data.device)
        return data



class UniformSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(torch.randperm(self.num_samples).tolist())

    def __len__(self):
        return self.num_samples



class ImbalancedSampler(Sampler):
    def __init__(self, train_y, imb_ratio):
        self.train_y = train_y
        self.imb_ratio = imb_ratio
        
        class_counts = np.bincount(self.train_y)
        class_ranks = np.argsort(np.argsort(class_counts))
        
        max_rank = class_ranks.max()
        sampling_probs = np.zeros_like(self.train_y, dtype=float)
        
        for class_index in range(len(class_counts)):
            rank = class_ranks[class_index]
            prob = (1 - (rank / max_rank)) * (imb_ratio - 1) + 1
            sampling_probs[self.train_y == class_index] = prob
        
        self.probabilities = sampling_probs / sampling_probs.sum()

    def __iter__(self):
        return iter(torch.multinomial(torch.tensor(self.probabilities), len(self.train_y), replacement=True).tolist())

    def __len__(self):
        return len(self.train_y)



class TemporallyCorrelatedSampler(Sampler):
    def __init__(self, test_y, alpha=1.0, window_size=5, epsilon=1e-6):
        self.test_y = test_y
        self.num_classes = len(np.unique(self.test_y))
        self.alpha = alpha  # Dirichlet distribution parameter
        self.window_size = window_size  # Window size for temporal correlation
        self.epsilon = epsilon  # Smoothing parameter for Dirichlet distribution

        # Start with a uniform distribution over labels
        self.current_probs = np.ones(self.num_classes) / self.num_classes

    def __iter__(self):
        sampled_indices = []
        for i in range(len(self.test_y)):
            # Sample a new probability distribution over labels using a Dirichlet distribution
            dirichlet_dist = dist.Dirichlet(torch.tensor(self.current_probs * self.alpha, dtype=torch.float32))
            sampled_probs = dirichlet_dist.sample().numpy()
            
            # Avoid zero probabilities
            sampled_probs = np.clip(sampled_probs, a_min=self.epsilon, a_max=None)
            sampled_probs = sampled_probs / sampled_probs.sum()
            
            # Sample a label based on the sampled_probs
            sampled_label = np.random.choice(np.arange(self.num_classes), p=sampled_probs)
            
            # Find an index of that label in the original dataset
            possible_indices = np.where(self.test_y == sampled_label)[0]
            sampled_index = np.random.choice(possible_indices)
            sampled_indices.append(sampled_index)
            
            # Update the probabilities based on the recent history (temporal correlation)
            if len(sampled_indices) >= self.window_size:
                recent_labels = self.test_y[sampled_indices[-self.window_size:]]
                label_counts = np.bincount(recent_labels, minlength=self.num_classes)
                self.current_probs = label_counts / label_counts.sum()
                
                # Avoid zero probabilities in updated current_probs
                self.current_probs = np.clip(self.current_probs, a_min=self.epsilon, a_max=None)
                self.current_probs = self.current_probs / self.current_probs.sum()

        return iter(sampled_indices)

    def __len__(self):
        return len(self.test_y)



class InverseLikelihoodSampler(Sampler):
    def __init__(self, likelihoods):
        self.likelihoods = torch.tensor(likelihoods, dtype=torch.float32)
        self.inverse_likelihoods = 1 / self.likelihoods
        self.probabilities = self.inverse_likelihoods / self.inverse_likelihoods.sum()

    def __iter__(self):
        return iter(torch.multinomial(self.probabilities, len(self.probabilities), replacement=True).tolist())

    def __len__(self):
        return len(self.probabilities)