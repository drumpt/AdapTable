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
