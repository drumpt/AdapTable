import torch.utils.data
import time
import numpy as np
from utils.graph_embedding import get_mi_matrix, get_node2vec_embedding, get_nx_graph
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import torch_geometric
import torch
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from torch_sparse import SparseTensor
from copy import deepcopy
import torch.nn.functional as F


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, args, dataset, type=1, num_batches=100):
        self.cont_indices = [i for i in range(dataset.cont_dim)]
        self.feat = torch.FloatTensor(dataset.train_x).type(torch.float32).to(args.device)
        self.cls = torch.tensor(dataset.train_y).to(args.device)
        self.num_batches = len(self.feat) // args.test_batch_size * 2
        self.dataset = dataset

        if hasattr(dataset, 'input_one_hot_encoder'):
            self.cat_len_per_node = [len(category) for category in dataset.input_one_hot_encoder.categories_]
            self.cat_idx_per_node = []

            offset = len(self.cont_indices)
            for cat_len in self.cat_len_per_node:
                self.cat_idx_per_node.append([i for i in range(offset, offset + cat_len)])
                offset += cat_len

            self.max_cat_len = max(self.cat_len_per_node)
        else:
            self.cat_len_per_node = []
            self.cat_idx_per_node = []
            self.max_cat_len = 0

        self.args = args
        self.num_cls = dataset.train_y.shape[1]
        self.created_batches = []
        self.created_batches_cls = []
        self.mi_idx = self.cont_indices + self.cat_idx_per_node
        self.type = type
        print(f'type of dataset is : {type}')

        self.construct_correlated_batches()
        self.construct_graph_batches()

    def construct_correlated_batches(self):
        num_batches = self.num_batches
        data = self.feat
        labels = torch.argmax(self.cls, dim=1)
        batch_size = self.args.test_batch_size
        classes = torch.unique(labels, return_counts=False)

        for train_x, train_y in self.dataset.posttrain_loader:
            train_x, train_y = train_x.to(self.args.device), train_y.to(self.args.device)
            self.created_batches.append(train_x)
            self.created_batches_cls.append(train_y)


        # random permute cont indices:
        # random_cont_indices = np.random.permutation(self.cont_indices)
        # random_cat_indices = np.random.permutation(self.cat_idx_per_node)
        #
        # for cat_idx in random_cat_indices:
        #     # sort features by cat_idx
        #     feat = self.feat.clone()
        #     cls = self.cls.clone()
        #
        #     # # add gaussian noise to the continuous features
        #     # feat[:, self.cont_indices] += torch.randn(feat[:, self.cont_indices].shape).to(self.args.device) * np.random.uniform(0, 0.1)
        #
        #     sorted_feat = feat[torch.argsort(torch.argmax(self.feat[:, cat_idx], dim=-1))]
        #     sorted_cls = cls[torch.argsort(torch.argmax(self.feat[:, cat_idx], dim=-1))]
        #
        #     # split features by cat_idx
        #     split_feat = torch.split(sorted_feat, self.args.test_batch_size)
        #     split_cls = torch.split(sorted_cls, self.args.test_batch_size)
        #
        #     if len(sorted_feat) % self.args.test_batch_size != 0:
        #         split_feat = split_feat[:-1]
        #         split_cls = split_cls[:-1]
        #     self.created_batches.extend(list(split_feat))
        #     self.created_batches_cls.extend(list(split_cls))
        #
        #     # random select at maximum 100 batches from self.created_batches
        #     if len(self.created_batches) > self.num_batches:
        #         break
        #
        # for cont_idx in random_cont_indices:
        #     # sort features by cont_idx
        #     feat = self.feat.clone()
        #     cls = self.cls.clone()
        #
        #     # # add gaussian noise to the continuous features
        #     # feat[:, self.cont_indices] += torch.randn(feat[:, self.cont_indices].shape).to(self.args.device) * np.random.uniform(0, 0.1)
        #
        #     sorted_feat = feat[torch.argsort(self.feat[:, cont_idx])]
        #     sorted_cls = cls[torch.argsort(self.feat[:, cont_idx])]
        #
        #     # split features by cont_idx
        #     split_feat = torch.split(sorted_feat, self.args.test_batch_size)
        #     split_cls = torch.split(sorted_cls, self.args.test_batch_size)
        #
        #     if len(sorted_feat) % self.args.test_batch_size != 0:
        #         split_feat = split_feat[:-1]
        #         split_cls = split_cls[:-1]
        #
        #     # TODO: add robustness to the batch - ex. Gaussian noise
        #     self.created_batches.extend(list(split_feat))
        #     self.created_batches_cls.extend(list(split_cls))
        #
        #     # random select at maximum 100 batches from self.created_batches
        #     if len(self.created_batches) > self.num_batches:
        #         break


        # # random select batches
        # if len(self.created_batches) > self.num_batches:
        #     idx = np.random.permutation(len(self.created_batches))
        #     self.created_batches = [self.created_batches[i] for i in idx[:self.num_batches]]
        #     self.created_batches_cls = [self.created_batches_cls[i] for i in idx[:self.num_batches]]

        # random permutation
        idx = np.random.permutation(len(self.created_batches))
        self.created_batches = [self.created_batches[i] for i in idx]
        self.created_batches_cls = [self.created_batches_cls[i] for i in idx]

        for idx, batch_cls in enumerate(self.created_batches_cls):
            np_cls = torch.argmax(batch_cls, dim=1).cpu().numpy()
            print(f'cls distribution of batch [{idx}/{len(self.created_batches_cls)}], {np.unique(np_cls, return_counts=True)}')


    def construct_graph_batches(self):
        # create graph batches
        self.created_graph_batches = []

        for batch_idx, batch in enumerate(self.created_batches):
            print(f'currently : {batch_idx} / {len(self.created_batches)}')
            mi_matrix, numerical_node_feat, categorical_node_feat = GraphDataset.get_features(
                args=self.args, batch=batch, mi_idx=self.mi_idx, type=self.type
            )
            num_nodes = mi_matrix.shape[0]
            edge_index, edge_weights = torch_geometric.utils.sparse.dense_to_sparse(mi_matrix)
            # print('max of edge_weights : ', edge_weights.max())
            # print('min of edge_weights : ', edge_weights.min())
            adj_t = SparseTensor.from_edge_index(edge_index, edge_weights, sparse_sizes=(num_nodes, num_nodes))

            graph_data = Data(num_x=numerical_node_feat, cat_x=categorical_node_feat, edge_index=adj_t,
                              edge_weights=edge_weights)
            self.created_graph_batches.append(graph_data)


    @staticmethod
    def calculate_mutual_information(args, arr1, arr2, is_arr1_categorical, is_arr2_categorical):

        if is_arr1_categorical and is_arr2_categorical:
            from sklearn.metrics import mutual_info_score
            return torch.tensor(
                mutual_info_score(arr1.detach().cpu().numpy(),
                                  arr2.detach().cpu().numpy())
            ).to(args.device).float()
        elif is_arr1_categorical:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            return torch.tensor(
                mutual_info_regression(arr1.unsqueeze(1).detach().cpu().numpy(),
                                       arr2.unsqueeze(1).detach().cpu().numpy())
            ).to(args.device).float()
        elif is_arr2_categorical:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            return torch.tensor(
                mutual_info_regression(arr2.unsqueeze(1).detach().cpu().numpy(),
                                       arr1.unsqueeze(1).detach().cpu().numpy())
            ).to(args.device).float()
        else:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            return torch.tensor(
                mutual_info_regression(arr1.unsqueeze(1).detach().cpu().numpy(),
                                        arr2.unsqueeze(1).detach().cpu().numpy())
            ).to(args.device).float()

    # @staticmethod
    # def calculate_correlation_coeff(args, arr1, arr2, is_arr1_categorical, is_arr2_categorical):


    def __len__(self):
        return len(self.dataset)


    @staticmethod
    def create_test_graph(args, dataset, batch, type=1):
        # set variables
        cont_indices = [i for i in range(dataset.cont_dim)]
        if hasattr(dataset, 'input_one_hot_encoder'):
            cat_len_per_node = [len(category) for category in dataset.input_one_hot_encoder.categories_]
            cat_idx_per_node = []

            offset = len(cont_indices)
            for cat_len in cat_len_per_node:
                cat_idx_per_node.append([i for i in range(offset, offset + cat_len)])
                offset += cat_len

            max_cat_len = max(cat_len_per_node)
        else:
            cat_len_per_node = []
            cat_idx_per_node = []

        mi_idx = cont_indices + cat_idx_per_node

        mi_matrix, numerical_node_feat, categorical_node_feat = GraphDataset.get_features(
            args=args, batch=batch, mi_idx=mi_idx, type=type
        )

        num_nodes = mi_matrix.shape[0]
        edge_index, edge_weights = torch_geometric.utils.sparse.dense_to_sparse(mi_matrix)
        adj_t = SparseTensor.from_edge_index(edge_index, edge_weights, sparse_sizes=(num_nodes, num_nodes))
        graph_data = Data(num_x=numerical_node_feat, cat_x=categorical_node_feat, edge_index=adj_t, edge_weights=edge_weights)

        return graph_data

    @staticmethod
    def get_mi_matrix(args, batch, idx_lists):

        mi_matrix = torch.zeros((len(idx_lists), len(idx_lists))).float().to(args.device)
        for idx1, i in enumerate(idx_lists):
            for idx2, j in enumerate(idx_lists):
                if i == j:
                    continue
                is_a_cat, is_b_cat = False, False
                if isinstance(i, list) and isinstance(j, list):
                    a = torch.argmax(batch[:, i], dim=1)
                    b = torch.argmax(batch[:, j], dim=1)
                    is_a_cat, is_b_cat = True, True
                elif isinstance(i, list):
                    a = torch.argmax(batch[:, i], dim=1)
                    b = batch[:, j]
                    is_a_cat = True
                elif isinstance(j, list):
                    a = batch[:, i]
                    b = torch.argmax(batch[:, j], dim=1)
                    is_b_cat = True
                else:
                    a = batch[:, i]
                    b = batch[:, j]

                mi_matrix[idx1, idx2] = GraphDataset.calculate_mutual_information(args, a, b, is_a_cat, is_b_cat)

    @staticmethod
    def get_correlation_matrix(args, batch, idx_lists):
        new_batch = torch.zeros((batch.shape[0], len(idx_lists))).float().to(args.device)
        for idx, i in enumerate(idx_lists):
            if isinstance(i, list):
                new_batch[:, idx] = torch.argmax(batch[:, i], dim=1)
            else:
                new_batch[:, idx] = batch[:, i]

        matrix = torch.corrcoef(new_batch.T) - torch.eye(len(idx_lists)).to(args.device)

        if torch.isnan(matrix).any():
            # nan in cases where variance is 0
            # print('nan in matrix')
            matrix = torch.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=0.0)

        # GCNConv cannot handle negaive weights
        matrix = torch.abs(matrix)

        # make lower-80% weights to zero
        # matrix = torch.where(matrix < torch.quantile(matrix, 0.6), torch.zeros_like(matrix), torch.ones_like(matrix))


        # print('maximum : ', torch.max(matrix))
        # print('minimum : ', torch.min(matrix))
        return matrix

    @staticmethod
    def get_stacked_distribution_features(args, mi_idx, batch):
        numerical_node_feat = []
        categorical_node_feat = []

        max_cat_len = 0
        for idx, i in enumerate(mi_idx):
            if isinstance(i, list):
                max_cat_len = max(max_cat_len, len(i))

        for idx, i in enumerate(mi_idx):
            if isinstance(i, list):
                prob_of_categories = torch.zeros(max_cat_len)
                prob_init = torch.sum(batch[:, i], dim=0) / batch.shape[0]
                prob_of_categories[:len(prob_init)] = prob_init
                categorical_node_feat.append(prob_of_categories.to(args.device))
            else:
                numerical_node_feat.append(
                    torch.stack([torch.mean(batch[:, i]), torch.std(batch[:, i])]).to(args.device))

        numerical_node_feat = torch.stack(numerical_node_feat).to(args.device)
        if len(categorical_node_feat) > 0:
            categorical_node_feat = torch.stack(categorical_node_feat).to(args.device)
        else:
            categorical_node_feat = torch.tensor([]).to(args.device)

        return numerical_node_feat, categorical_node_feat

    @staticmethod
    def get_stacked_shift_features(args, mi_idx, batch):
        numerical_node_feat = []
        categorical_node_feat = []

        max_cat_len = 0
        for idx, i in enumerate(mi_idx):
            if isinstance(i, list):
                max_cat_len = max(max_cat_len, len(i))

        for idx, i in enumerate(mi_idx):
            if isinstance(i, list):
                cat_batch = torch.argmax(batch[:, i], dim=-1).float()
                batch_mean = torch.mean(cat_batch, dim=0)
                batch_std = torch.std(cat_batch, dim=0) + 1e-6

                categorical_node_feat.append(
                    ((cat_batch - batch_mean) / batch_std).to(args.device)
                )
            else:
                num_batch = batch[:, i].float()
                batch_mean = torch.mean(num_batch, dim=0)
                batch_std = torch.std(num_batch, dim=0) + 1e-6

                numerical_node_feat.append(
                    ((num_batch - batch_mean) / batch_std).to(args.device)
                )

        numerical_node_feat = torch.stack(numerical_node_feat).to(args.device)
        categorical_node_feat = torch.stack(categorical_node_feat).to(args.device)

        # normalize
        numerical_node_feat = F.normalize(numerical_node_feat, p=2, dim=1)
        categorical_node_feat = F.normalize(categorical_node_feat, p=2, dim=1)

        return numerical_node_feat, categorical_node_feat

    @staticmethod
    def get_stacked_renormalized_features(args, mi_idx, batch):
        numerical_node_feat = []
        categorical_node_feat = []

        max_cat_len = 0
        for idx, i in enumerate(mi_idx):
            if isinstance(i, list):
                max_cat_len = max(max_cat_len, len(i))

        for idx, i in enumerate(mi_idx):
            if isinstance(i, list):
                cat_batch = batch[:, i].float()
                batch_mean = torch.mean(cat_batch, dim=0)
                cat_features = cat_batch - batch_mean
                # fill in zeros to match max_cat_len
                cat_features = torch.cat([cat_features, torch.zeros(len(cat_features), max_cat_len - len(i)).to(args.device)], dim=1)
                categorical_node_feat.append(cat_features)
            else:
                num_batch = batch[:, i]
                batch_mean = torch.mean(num_batch, dim=0)
                num_features = num_batch - batch_mean
                numerical_node_feat.append(num_features)

        if numerical_node_feat:
            numerical_node_feat = torch.stack(numerical_node_feat).to(args.device)
            numerical_node_feat = F.normalize(numerical_node_feat, p=2, dim=1)
        if categorical_node_feat:
            categorical_node_feat = torch.stack(categorical_node_feat).to(args.device)
            categorical_node_feat = F.normalize(categorical_node_feat, p=2, dim=1)

        return numerical_node_feat, categorical_node_feat

    @staticmethod
    def get_features(type, args, mi_idx, batch):
        if type in [0]:  # type 0: mutual info. as adj. matrix + node as cols + node feature as stats
            mi_matrix = GraphDataset.get_mi_matrix(args=args, batch=batch, idx_lists=mi_idx)
            numerical_node_feat, categorical_node_feat = GraphDataset.get_stacked_distribution_features(
                args=args, batch=batch, mi_idx=mi_idx
            )
            # print('type 0')
        elif type in [1]:  # type 1: correlation info. as adj. matrix + node as cols  + node feature as stats
            mi_matrix = GraphDataset.get_correlation_matrix(args=args, batch=batch, idx_lists=mi_idx)
            numerical_node_feat, categorical_node_feat = GraphDataset.get_stacked_distribution_features(
                args=args, batch=batch, mi_idx=mi_idx
            )
            # print('type 1')
        elif type in [2]:  # correlation info. as adj. matrix + node as cols  + shift as stats
            mi_matrix = GraphDataset.get_correlation_matrix(args=args, batch=batch, idx_lists=mi_idx)
            numerical_node_feat, categorical_node_feat = GraphDataset.get_stacked_shift_features(
                args=args, batch=batch, mi_idx=mi_idx
            )
            # print('type 2')
        elif type in [3]:
            mi_matrix = GraphDataset.get_correlation_matrix(args=args, batch=batch, idx_lists=mi_idx)
            numerical_node_feat, categorical_node_feat = GraphDataset.get_stacked_renormalized_features(
                args=args, batch=batch, mi_idx=mi_idx
            )
            # print('type 3')
        else:
            raise NotImplementedError

        return mi_matrix, numerical_node_feat, categorical_node_feat




