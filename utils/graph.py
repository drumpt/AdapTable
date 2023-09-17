import numpy as np

from utils.graph_embedding import get_mi_matrix, get_node2vec_embedding, get_nx_graph
import torch
from torch_geometric.utils import from_networkx
from model.graph import GraphNet
from utils.graph_embedding import get_batched_mi_matrix, get_batched_graph
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

def get_pretrained_graphnet_rowwise(args, dataset, source_model):
    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
        args.device)
    valid_x, valid_y = torch.tensor(dataset.valid_x).float().to(args.device), torch.tensor(dataset.valid_y).float().to(
        args.device)
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
        args.device)

    # construct gnn
    gnn = GraphNet(
        num_features=dataset.train_x.shape[1],
        num_classes=dataset.train_y.shape[1]
    ).to(args.device)

    gnn.requires_grad_(True)
    gnn.train()
    optimizer = torch.optim.AdamW(gnn.parameters(), lr=0.001)

    valid_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y), batch_size=args.test_batch_size, shuffle=True, drop_last=True)

    best_gnn = deepcopy(gnn)
    best_valid_loss = np.inf

    for epoch in range(20):
        loss_total = 0
        train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                                       batch_size=args.test_batch_size, shuffle=True, drop_last=True)
        for batched_train_x, batchced_train_y in tqdm(train_dataloader):
            batched_train_x, batchced_train_y = batched_train_x.to(args.device), batchced_train_y.to(args.device)

            # construct adjacency matrxi via cossim between row's features
            with torch.no_grad():
                batched_feat_x = source_model.get_feature(batched_train_x)
                normalized_batched_feat_x = F.normalize(batched_feat_x, dim=-1)
                adj_matrix = torch.matmul(normalized_batched_feat_x, normalized_batched_feat_x.T)

            # construct graph
            batched_graph = from_networkx(get_batched_graph(adj_matrix))

            # fill in x values - the actual input features
            batched_graph.x = batched_train_x

            # cast to device
            batched_graph = batched_graph.to(args.device)

            vanilla_out = source_model(batched_train_x).detach()
            gnn_out = gnn(batched_graph)
            estimated_y = F.softmax(vanilla_out + gnn_out, dim=-1)

            loss = F.cross_entropy(estimated_y, torch.argmax(batchced_train_y, dim=-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
        print(f'epoch {epoch} loss is {loss_total / len(train_dataloader)}')

        if epoch % 1 == 0:
            valid_loss = 0
            with torch.no_grad():
                for valid_x, valid_y in tqdm(valid_dataloader):
                    # log loss on validation set
                    vanilla_out = source_model(valid_x).detach()

                    batched_feat_x = source_model.get_feature(valid_x)
                    normalized_batched_feat_x = F.normalize(batched_feat_x, dim=-1)
                    adj_matrix = torch.matmul(normalized_batched_feat_x, normalized_batched_feat_x.T)

                    # construct graph
                    batched_graph = from_networkx(get_batched_graph(adj_matrix))

                    # fill in x values - the actual input features
                    batched_graph.x = valid_x

                    # cast to device
                    batched_graph = batched_graph.to(args.device)

                    gnn_out = gnn(batched_graph)

                    estimated_y = F.softmax(vanilla_out + gnn_out, dim=-1)
                    valid_loss += F.cross_entropy(estimated_y, torch.argmax(valid_y, dim=-1))

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_gnn = deepcopy(gnn)

                print(f'epoch {epoch} validation loss is {valid_loss.item() / len(valid_dataloader)}')

    return best_gnn

def get_graphnet_out_rowwise(args, data, source_model, gnn):
    # construct graph

    # construct adj matrix via features
    with torch.no_grad():
        feat_x = source_model.get_feature(data)
        normalized_feat_x = F.normalize(feat_x, dim=-1)
        adj_matrix = torch.matmul(normalized_feat_x, normalized_feat_x.T)

    # construct graph
    graph = from_networkx(get_batched_graph(adj_matrix))
    graph.x = data
    graph = graph.to(args.device)

    # get gnn out
    gnn_out = gnn(graph)

    # get vanilla out
    vanilla_out = F.softmax(source_model(data), dim=-1).detach()

    # get estimated out
    estimated_out = F.softmax(vanilla_out + gnn_out, dim=-1)

    return estimated_out








def get_pretrained_graphnet_columnwise(args, dataset, source_model):
    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
        args.device)

    tot_train_x = train_x
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
        args.device)

    from data.graph_data import GraphDataset
    train_graph_dataset = GraphDataset(args, dataset)

    # construct gnn
    gnn = GraphNet(
        num_features=2,
        num_classes=dataset.train_y.shape[1],
        cat_cls_len=train_graph_dataset.cat_len_per_node,
        cont_len=len(train_graph_dataset.cont_indices)
    ).to(args.device)
    gnn.requires_grad_(True)
    gnn.train()

    optimizer = torch.optim.AdamW(gnn.parameters(), lr=0.01)


    for epoch in range(100):
        loss_total = 0
        for batched_train_x, batched_graph, batched_train_y in tqdm(zip(train_graph_dataset.created_batches, train_graph_dataset.created_graph_batches, train_graph_dataset.created_batches_cls)):

            batched_train_x, batched_train_y = batched_train_x.to(args.device).float(), batched_train_y.to(args.device).float()
            batched_graph = batched_graph.to(args.device)

            vanilla_out = source_model(batched_train_x).detach() # currently vanilla outs are logits
            gnn_out = gnn(batched_graph) # currently gnn outs are logits
            estimated_y = vanilla_out + gnn_out

            loss = F.cross_entropy(estimated_y, torch.argmax(batched_train_y, dim=-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

        print(f'epoch {epoch} loss is {loss_total / len(train_graph_dataset.created_batches)}')

    return gnn

def get_graphnet_out_columnwise(args, dataset, batch, source_model, gnn):
    from data.graph_data import GraphDataset
    test_graph_data = GraphDataset.create_test_graph(args, dataset, batch)

    # get gnn out
    gnn_out = gnn(test_graph_data)
    print(f'gnn out is : {gnn_out[0]}')
    source_out = source_model(batch).detach()

    estimated_out = F.softmax(source_out + gnn_out, dim=-1)
    return estimated_out


