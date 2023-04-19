import os
from os import path

import openml
from openml import tasks, runs
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import torch



class Datasets():
    def __init__(self, args):
        if args.meta_dataset == "openml-cc18":
            dataset = OpenMLDatasets()
        elif args.meta_dataset == "tableshift":
            pass
        elif args.meta_dataset == "openml-regression":
            pass

        self.train_data = torch.utils.data.TensorDataset(torch.from_numpy(dataset.train_x), torch.from_numpy(dataset.train_y))
        self.valid_data = torch.utils.data.TensorDataset(torch.from_numpy(dataset.valid_x), torch.from_numpy(dataset.valid_y))
        # if args.meta_dataset != "tableshift": # tableshift requires natural shift from model uncertainty only
        #     dataset.test_x = self.get_corrupted_data(dataset.test_x, args.corruption_type, args.corruption_severity, args.imputation_method)
        self.test_data = torch.utils.data.TensorDataset(torch.from_numpy(dataset.test_x), torch.from_numpy(dataset.test_y))

        self.in_dim, self.out_dim = dataset.train_x.shape[-1], dataset.train_y.shape[-1]
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=args.train_batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_data, batch_size=args.train_batch_size, shuffle=True)
        self.test_loader =  torch.utils.data.DataLoader(self.test_data, batch_size=args.test_batch_size, shuffle=True)



class OpenMLDatasets():
    def __init__(self, args):
        benchmark_list_path = "data/OpenML-CC18/benchmark_list.csv"
        if not os.path.exists(benchmark_list_path):
            benchmark = openml.study.get_suite("OpenML-CC18")
            benchmark_df = tasks.list_tasks(output_format="dataframe", task_id=benchmark.tasks)
            benchmark_df.to_csv(benchmark_list_path)
        else:
            benchmark_df = pd.read_csv(benchmark_list_path)
        target_feature = benchmark_df[benchmark_df["name"] == args.name].iloc[0]["target_feature"]
        
        dataset = openml.datasets.get_dataset(args.name)
        x, y, cat_indicator, _ = dataset.get_data(target=target_feature, dataset_format="dataframe")
        y = np.array(y).reshape(-1, 1)

        # train/valid/test split
        train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4, random_state=42)
        valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y, test_size=0.5, random_state=42)

        # data preprocessing (normalization and one-hot encoding)
        cat_indices = np.argwhere(np.array(cat_indicator) == True).flatten()
        cont_indices = np.array(sorted(set(np.arange(x.shape[1] - 1)).difference(set(cat_indices))))
        if len(cont_indices):
            self.input_scaler = getattr(sklearn.preprocessing, args.normalizer)()
            self.input_scaler.fit(np.concatenate([train_x.iloc[:, cont_indices], valid_x.iloc[:, cont_indices]], axis=0))
            train_cont_x = self.input_scaler.transform(train_x.iloc[:, cont_indices])
            valid_cont_x = self.input_scaler.transform(valid_x.iloc[:, cont_indices])
            test_cont_x = self.input_scaler.transform(get_corrupted_data((test_x.iloc[:, cont_indices], )))
        else:
            train_cont_x, valid_cont_x, test_cont_x = np.array([]), np.array([]), np.array([])
        if len(cat_indices):
            self.input_one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.input_one_hot_encoder.fit(np.concatenate([train_x.iloc[:, cat_indices], valid_x.iloc[:, cat_indices]], axis=0))
            train_cat_x = self.input_one_hot_encoder.transform(train_x.iloc[:, cat_indices])
            valid_cat_x = self.input_one_hot_encoder.transform(valid_x.iloc[:, cat_indices])
            test_cat_x = self.input_one_hot_encoder.transform(test_x.iloc[:, cat_indices])
        else:
            train_cat_x, valid_cat_x, test_cat_x = np.array([]), np.array([]), np.array([])
        self.train_x = np.concatenate([
            train_cont_x if len(cont_indices) else train_cont_x.reshape(train_cat_x.shape[0], 0),
            train_cat_x if len(cat_indices) else train_cat_x.reshape(train_cont_x.shape[0], 0)
            ], axis=-1
        )
        self.valid_x = np.concatenate([
            valid_cont_x if len(cont_indices) else valid_cont_x.reshape(valid_cat_x.shape[0], 0),
            valid_cat_x if len(cat_indices) else valid_cat_x.reshape(valid_cont_x.shape[0], 0)
            ], axis=-1
        )
        self.test_x = np.concatenate([
            test_cont_x if len(cont_indices) else test_cont_x.reshape(test_cat_x.shape[0], 0),
            test_cat_x if len(cat_indices) else test_cat_x.reshape(test_cont_x.shape[0], 0)
            ], axis=-1
        )

        # TODO: implement normalization for regression dataset
        self.output_one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.output_one_hot_encoder.fit(np.concatenate([train_y, valid_y], axis=0))
        self.train_y = self.output_one_hot_encoder.transform(train_y)
        self.valid_y = self.output_one_hot_encoder.transform(valid_y)
        self.test_y = self.output_one_hot_encoder.transform(test_y)



# class UCIDatasets():
#     def __init__(self, args, name, data_path):
#         self.datasets = {
#             "wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", # classification
#             "adult": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", # classification
#             "breast": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data", # classification

#             "concrete": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls", # regression
#             "housing": "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", # regression
#             "energy": "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx", # regression
#             "power": "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip", # regression
#             "yacht": "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data", # regression
#         }
#         self.args = args
#         self.name = name
#         self.data_path = data_path
#         self.load_dataset()


#     def load_dataset(self):
#         if self.name not in self.datasets:
#             raise Exception("Unknown dataset!")
#         if not path.exists(self.data_path + "UCI"):
#             os.mkdir(self.data_path + "UCI")

#         url = self.datasets[self.name]
#         file_name = url.split('/')[-1]
#         if not path.exists(os.path.join(self.data_path, "UCI", file_name)):
#             urllib.request.urlretrieve(self.datasets[self.name], os.path.join(self.data_path, "UCI", file_name))

#         # get dataset
#         if self.name == "wine":
#             data = pd.read_csv(os.path.join(self.data_path, 'UCI/winequality-red.csv'), header=1, delimiter=';').values
#             cat_indices = np.array([])
#             task = "classification"
#         elif self.name == "adult":
#             data = pd.read_csv(os.path.join(self.data_path, 'UCI/adult.data'), header=0, delimiter=", ").values
#             cat_indices = np.array([1, 3, 5, 6, 7, 8, 9, 13])
#             task = "classification"
#         elif self.name == "breast":
#             data = pd.read_csv(os.path.join(self.data_path, 'UCI/breast.data'), header=0, delimiter="\s+").values
#             task = "classification"
#         elif self.name == "concrete":
#             data = pd.read_excel(os.path.join(self.data_path, 'UCI/Concrete_Data.xls'), header=0).values
#             task = "regression"
#         elif self.name == "housing":
#             data = pd.read_csv(os.path.join(self.data_path, 'UCI/housing.data'), header=0, delimiter="\s+").values
#             task = "regression"
#         elif self.name == "energy":
#             data = pd.read_excel(os.path.join(self.data_path, 'UCI/ENB2012_data.xlsx'), header=0).values
#             task = "regression"
#         elif self.name == "power":
#             zipfile.ZipFile(os.path.join(self.data_path, "UCI/CCPP.zip")).extractall(os.path.join(self.data_path, "UCI/CCPP/"))
#             data = pd.read_excel(os.path.join(self.data_path, "UCI/CCPP.zip"), header=0).values
#             task = "regression"
#         elif self.name == "yacht":
#             data = pd.read_csv(os.path.join(self.data_path, 'UCI/yacht_hydrodynamics.data'), header=1, delimiter='\s+').values
#             task = "regression"

#         if task == "classification":
#             self.output_one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='error')
#             output = self.output_one_hot_encoder.fit_transform(np.expand_dims(data[:, -1], axis=1))
#             train_x, valid_x, train_y, valid_y = train_test_split(data[:, :-1], output, test_size=0.2, random_state=42, shuffle=True)
#             valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y, test_size=0.2, random_state=42, shuffle=True)
#         else:
#             # train/valid/test split
#             self.in_dim, self.out_dim = data.shape[1] - 1, 1

#             train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
#             valid_data, test_data = train_test_split(data, test_size=0.5, random_state=42, shuffle=True)
#             train_x, valid_x, test_x = train_data[:, :self.in_dim], valid_data[:, :self.in_dim], test_data[:, :self.in_dim]
#             train_y, valid_y, test_y = train_data[:, self.in_dim:], valid_data[:, self.in_dim:], test_data[:, self.in_dim:]

#         # normalize w.r.t. train dataset
#         cont_indices = np.array(sorted(set(np.arange(data.shape[1] - 1)).difference(set(cat_indices))))
#         self.input_scaler = StandardScaler()

#         train_cont_x = self.input_scaler.fit_transform(train_x[:, cont_indices])
#         valid_cont_x = self.input_scaler.transform(valid_x[:, cont_indices])
#         test_cont_x = self.input_scaler.transform(test_x[:, cont_indices])

#         if len(cat_indices):
#             self.input_one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#             train_x = np.concatenate([train_cont_x, self.input_one_hot_encoder.fit_transform(train_x[:, cat_indices])], axis=-1)
#             valid_x = np.concatenate([valid_cont_x, self.input_one_hot_encoder.transform(valid_x[:, cat_indices])], axis=-1)
#             test_x = np.concatenate([test_cont_x, self.input_one_hot_encoder.transform(test_x[:, cat_indices])], axis=-1)

#         if task == "regression":
#             self.output_scaler = StandardScaler()
#             train_y = self.output_scaler.fit_transform(train_y)
#             valid_y = self.output_scaler.transform(valid_y)
#             test_y = self.output_scaler.transform(test_y)

#         train_data = torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
#         valid_data = torch.utils.data.TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
#         test_data = torch.utils.data.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

#         self.in_dim, self.out_dim = train_x.shape[-1], train_y.shape[-1]

#         self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
#         self.valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
#         self.test_loader =  torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)



def get_corrupted_data(test_data, train_data, corruption_type, corruption_severity, imputation_method):
    if corruption_type == "Gaussian":
        test_data += corruption_severity * torch.randn_like(test_data, device=test_data.device)

    elif corruption_type in ["random_crop", "column_drop", "column_block_drop"]:
        assert 0 <= corruption_severity <= 1

        if corruption_type == "random_drop":
            mask = (torch.zeros_like(test_data).to(test_data.device).uniform_() > corruption_severity).int()
        elif corruption_type == "column_drop":
            mask = (torch.zeros_like(test_data[0]).to(test_data.device).uniform_() > corruption_severity).int().unsqueeze(0)
        elif corruption_type == "column_block_drop":
            start_idx = np.random.randint(0, int(data[0].shape[-1] * (1 - corruption_severity)) + 1)
            end_idx = start_idx + int(data[0].shape[-1] * corruption_severity)
            mask = torch.tensor([1 if start_idx <= idx <= end_idx else 0 for idx in range(len(test_data[0]))]).unsqueeze(0)
        imputed_data = get_imputed_data(test_data, train_data, imputation_method)
        test_data = mask * test_data + (1 - mask) * imputed_data

    elif corruption_type in ["mean_shift", "std_shift", "mean_std_shift"]:
        if corruption_type == "mean_shift":
            mean = 2 * torch.rand(1).to(data.device) - 1
            data += mean
        elif corruption_type == "std_shift":
            std = 1.5 * torch.rand(1).to(data.device) + 0.5
            data *= std
        elif corruption_type == "mean_std_shift":
            mean = 2 * torch.rand(1).to(data.device) - 1        
            std = 1.5 * torch.rand(1).to(data.device) + 0.5
            data = std * data + mean
    return test_data


def get_imputed_data(test_data, train_data, imputation_method):
    return test_data