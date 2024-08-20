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


def torch_corrcoef(x):
    # Ensure input is 2D
    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() != 2:
        raise ValueError("Input must be a 1D or 2D tensor")

    # Compute the covariance matrix
    mean_x = torch.mean(x, dim=1, keepdim=True)
    xm = x - mean_x
    c = xm @ xm.t() / (x.size(1) - 1)

    # Compute the standard deviations
    d = torch.diag(c)
    stddev = torch.sqrt(d)

    # Compute the correlation coefficient matrix
    corr_matrix = c / (stddev[:, None] * stddev[None, :])

    # Clamp values to the range [-1, 1] to avoid floating point errors
    corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)

    return corr_matrix

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, args, dataset):
        self.cont_indices = [i for i in range(dataset.cont_dim)]
        self.feat = torch.FloatTensor(dataset.train_x).type(torch.float32).to(args.device)
        self.cls = torch.tensor(dataset.train_y).to(args.device)
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

        self.construct_correlated_batches()
        self.construct_graph_batches()



    def construct_correlated_batches(self):
        for train_x, train_y in self.dataset.posttrain_loader:
            train_x, train_y = train_x.to(self.args.device), train_y.to(self.args.device)
            self.created_batches.append(train_x)
            self.created_batches_cls.append(train_y)
        # random permutation
        idx = np.random.permutation(len(self.created_batches))
        self.created_batches = [self.created_batches[i] for i in idx]
        self.created_batches_cls = [self.created_batches_cls[i] for i in idx]

    def construct_graph_batches(self):
        # create graph batches
        self.created_graph_batches = []

        for batch_idx, batch in enumerate(self.created_batches):
            mi_matrix, numerical_node_feat, categorical_node_feat = GraphDataset.get_features(
                args=self.args, batch=batch, mi_idx=self.mi_idx, dataset=self.dataset,
            )
            num_nodes = mi_matrix.shape[0]
            edge_index, edge_weights = torch_geometric.utils.sparse.dense_to_sparse(mi_matrix)
            adj_t = SparseTensor.from_edge_index(edge_index, edge_weights, sparse_sizes=(num_nodes, num_nodes))

            graph_data = Data(num_x=numerical_node_feat, cat_x=categorical_node_feat, edge_index=adj_t,
                              edge_weights=edge_weights)
            self.created_graph_batches.append(graph_data)

    def __len__(self):
        return len(self.dataset)


    @staticmethod
    def create_test_graph(args, dataset, batch, mi_matrix=None):
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
            args=args, batch=batch, mi_idx=mi_idx, dataset=dataset, mi_matrix=mi_matrix
        )

        num_nodes = mi_matrix.shape[0]
        edge_index, edge_weights = torch_geometric.utils.sparse.dense_to_sparse(mi_matrix)
        adj_t = SparseTensor.from_edge_index(edge_index, edge_weights, sparse_sizes=(num_nodes, num_nodes))
        graph_data = Data(num_x=numerical_node_feat, cat_x=categorical_node_feat, edge_index=adj_t, edge_weights=edge_weights)

        return graph_data

    @staticmethod
    def get_correlation_matrix(args, batch, idx_lists):
        new_batch = torch.zeros((batch.shape[0], len(idx_lists))).float().to(args.device)
        for idx, i in enumerate(idx_lists):
            if isinstance(i, list):
                new_batch[:, idx] = torch.argmax(batch[:, i], dim=1)
            else:
                new_batch[:, idx] = batch[:, i]

        matrix = torch_corrcoef(new_batch.T) - torch.eye(len(idx_lists)).to(args.device)

        if torch.isnan(matrix).any():
            matrix = torch.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=0.0)

        # GCNConv cannot handle negaive weights
        matrix = torch.abs(matrix)
        return matrix

    @staticmethod
    def get_stacked_renormalized_features(args, mi_idx, batch, dataset):
        numerical_node_feat = []
        categorical_node_feat = []

        # torch_train_dataset = torch.tensor(dataset.train_x).to(args.device)
        torch_train_dataset = torch.tensor(dataset.train_x)
        torch_train_dataset_mean_dict = dict()        
        for idx, i in enumerate(mi_idx):
            if isinstance(i, list):
                cat_mean = torch.mean(torch_train_dataset[:, i].float(), dim=0)
                torch_train_dataset_mean_dict[repr(i)] = cat_mean
            else:
                num_mean = torch.mean(torch_train_dataset[:, i], dim=0)
                torch_train_dataset_mean_dict[i] = num_mean

        max_cat_len = 0
        for idx, i in enumerate(mi_idx):
            if isinstance(i, list):
                max_cat_len = max(max_cat_len, len(i))

        for idx, i in enumerate(mi_idx):
            if isinstance(i, list):
                cat_batch = batch[:, i].float()
                # train_cat_batch = torch_train_dataset[:, i].float()
                cat_features = cat_batch - torch_train_dataset_mean_dict[repr(i)].to(cat_batch.device)
                # fill in zeros to match max_cat_len
                cat_features = torch.cat([cat_features, torch.zeros(len(cat_features), max_cat_len - len(i)).to(args.device)], dim=1)
                categorical_node_feat.append(cat_features)
            else:
                num_batch = batch[:, i]
                # train_num_batch = torch_train_dataset[:, i]
                num_features = num_batch - torch_train_dataset_mean_dict[i].to(num_batch.device)
                numerical_node_feat.append(num_features)

        if numerical_node_feat:
            numerical_node_feat = torch.stack(numerical_node_feat).to(args.device)
        if categorical_node_feat:
            categorical_node_feat = torch.stack(categorical_node_feat).to(args.device)

        return numerical_node_feat, categorical_node_feat

    @staticmethod
    def get_features(args, mi_idx, batch, dataset=None, mi_matrix=None):
        # mi_matrix = GraphDataset.get_correlation_matrix(args=args, batch=batch, idx_lists=mi_idx)
        numerical_node_feat, categorical_node_feat = GraphDataset.get_stacked_renormalized_features(
            args=args, batch=batch, mi_idx=mi_idx, dataset=dataset
        )
        mi_matrix = torch.ones(len(mi_idx), len(mi_idx))

        return mi_matrix, numerical_node_feat, categorical_node_feat




