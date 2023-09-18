import numpy as np
import networkx as nx
import os
import torch
from node2vec import Node2Vec

from sklearn.feature_selection import mutual_info_regression


def get_mi_matrix(args, dataset, type='train'):

    data = dataset.train_x if type == 'train' else dataset.test_x

    # construct mi matrix
    if os.path.exists(f'./{args.dataset}_mi_matrix_{type}.npy'):
        mi_matrix = np.load(f'./{args.dataset}_mi_matrix_{type}.npy')
    else:
        mi_matrix = np.zeros((data.shape[1], data.shape[1]))

        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                if i == j:
                    continue
                mi_matrix[i, j] = mutual_info_regression(data[:, [i]], data[:, j])

        # save train mi matrix
        np.save(f'./{args.dataset}_mi_matrix_{type}.npy', mi_matrix)

    return mi_matrix

def get_batched_mi_matrix(args, data):
    def histogram2d(x, y, bins=10):
        # Create an array of bin edges
        bin_edges = torch.linspace(0, 1, bins + 1, device='cuda')

        # Compute the bin indices for each element in x and y
        bin_indices_x = (x.unsqueeze(1) >= bin_edges[:-1].unsqueeze(0)) & (x.unsqueeze(1) < bin_edges[1:].unsqueeze(0))
        bin_indices_y = (y.unsqueeze(1) >= bin_edges[:-1].unsqueeze(0)) & (y.unsqueeze(1) < bin_edges[1:].unsqueeze(0))

        # Flatten the last two dimensions
        bin_indices_x = bin_indices_x.view(bin_indices_x.size(0), -1)
        bin_indices_y = bin_indices_y.view(bin_indices_y.size(0), -1)

        # Compute the joint histogram
        joint_hist = torch.einsum('ij,ik->jk', bin_indices_x.float(), bin_indices_y.float())

        return joint_hist

    def mutual_information(x, y, bins=10):
        # Compute the joint histogram
        joint_hist = histogram2d(x, y, bins=bins)

        # Compute the marginal histograms
        x_hist = torch.histc(x, bins=bins)
        y_hist = torch.histc(y, bins=bins)

        # Compute the joint and marginal probabilities
        joint_prob = joint_hist / joint_hist.sum()
        x_prob = x_hist / x_hist.sum()
        y_prob = y_hist / y_hist.sum()

        # Compute the mutual information
        mi = torch.nansum(joint_prob * torch.log(joint_prob / (x_prob[:, None] * y_prob[None, :])))

        return mi

    if isinstance(data, torch.Tensor):
        data = data.to(args.device)
    elif isinstance(data, np.ndarray):
        data = torch.tensor(data).float().to(args.device)
    else:
        raise ValueError('data should be either numpy.ndarray or torch.Tensor')

    mi_matrix = torch.zeros((data.shape[1], data.shape[1])).to(args.device)

    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            if i == j:
                continue
            mi_matrix[i, j] = mutual_information(data[:, [i]], data[:, j])

    return mi_matrix

def get_nx_graph(mi_matrix):
    G = nx.Graph()

    if isinstance(mi_matrix, np.ndarray):
        pass
    elif isinstance(mi_matrix, torch.Tensor):
        mi_matrix = mi_matrix.detach().cpu().numpy()
    else:
        raise ValueError('mi_matrix should be either numpy.ndarray or torch.Tensor')

    for i in range(mi_matrix.shape[0]):
        for j in range(mi_matrix.shape[1]):
            if i == j:
                continue
            elif j >= i:
                G.add_edge(i, j, weight=mi_matrix[i, j])

    return G

def get_batched_graph(batched_mi_matrix):
    G = nx.Graph()

    if isinstance(batched_mi_matrix, np.ndarray):
        pass
    elif isinstance(batched_mi_matrix, torch.Tensor):
        mi_matrix = batched_mi_matrix.detach().cpu().numpy()
    else:
        raise ValueError('mi_matrix should be either numpy.ndarray or torch.Tensor')

    for i in range(batched_mi_matrix.shape[0]):
        for j in range(batched_mi_matrix.shape[1]):
            if i == j:
                continue
            elif j >= i:
                G.add_edge(i, j, weight=batched_mi_matrix[i, j])

    return G

def get_node2vec_embedding(args, mi_matrix, type='train'):
    G_train = nx.Graph()

    if os.path.exists(f'./{args.dataset}_node2vec_embedding_{type}.npy'):
        node_embeddings = np.load(f'./{args.dataset}_node2vec_embedding_{type}.npy', allow_pickle=True)
    else:
        if isinstance(mi_matrix, np.ndarray):
            pass
        elif isinstance(mi_matrix, torch.Tensor):
            mi_matrix = mi_matrix.detach().cpu().numpy()
        else:
            raise ValueError('mi_matrix should be either numpy.ndarray or torch.Tensor')

        for i in range(mi_matrix.shape[0]):
            for j in range(mi_matrix.shape[1]):
                if i == j:
                    continue
                elif j >= i:
                    G_train.add_edge(i, j, weight=mi_matrix[i, j])
        # Initialize Node2Vec model
        node2vec = Node2Vec(G_train, dimensions=64, walk_length=30, num_walks=100, workers=8, p=1, q=1, weight_key='weight')
        # Train Node2Vec model
        graphmodel = node2vec.fit(window=10, min_count=1)
        node_embeddings = graphmodel.wv.vectors
        # save train node2vec embedding
        np.save(f'./{args.dataset}_node2vec_embedding_{type}.npy', node_embeddings)

    return node_embeddings
