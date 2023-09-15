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

            vanilla_out = F.softmax(source_model(batched_train_x), dim=-1).detach()
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
                    vanilla_out = F.softmax(source_model(valid_x), dim=-1).detach()

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








def get_pretrained_graphnet(args, dataset, source_model):
    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
        args.device)

    tot_train_x = train_x
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
        args.device)


    mi_matrix_train = get_mi_matrix(args, dataset, 'train')
    mi_matrix_test = get_mi_matrix(args, dataset, 'test')

    mi_matrix_train = torch.tensor(mi_matrix_train).float().to(args.device)
    mi_matrix_test = torch.tensor(mi_matrix_test).float().to(args.device)

    graph_train_input = get_nx_graph(mi_matrix_train)
    graph_test_input = get_nx_graph(mi_matrix_test)

    with torch.no_grad():
        for i in range(len(graph_train_input.nodes)):
            with torch.no_grad():
                mask = torch.zeros_like(train_x[0])
                mask[i] = 1
                train_x_masked = train_x * mask
                feat_dim = source_model.get_feature(train_x_masked).shape[-1]
                graph_train_input.nodes[i]['x'] = torch.mean(source_model.get_feature(train_x_masked),
                                                             dim=0).detach()  # Replace num_features with the actual number of features per node

        for i in range(len(graph_test_input.nodes)):
            with torch.no_grad():
                mask = torch.zeros_like(train_x[0])
                mask[i] = 1
                test_x_masked = test_x * mask
                graph_test_input.nodes[i]['x'] = torch.mean(source_model.get_feature(test_x_masked),
                                                            dim=0).detach()  # Replace num_features with the actual number of features per node

    graph_train_input = from_networkx(graph_train_input)
    graph_test_input = from_networkx(graph_test_input)

    graph_train_input = graph_train_input.to(args.device)
    graph_test_input = graph_test_input.to(args.device)

    gnn = GraphNet(num_features=feat_dim*2, num_classes=dataset.train_y.shape[1]).to(args.device)
    gnn.requires_grad_(True)
    gnn.train()
    optimizer = torch.optim.AdamW(gnn.parameters(), lr=0.0001)
    source_model.requires_grad_(False)

    for epoch in range(100):
        loss_total = 0
        train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                                       batch_size=args.test_batch_size, shuffle=True, drop_last=True)
        for train_x, train_y in tqdm(train_dataloader):
            train_x, train_y = train_x.to(args.device), train_y.to(args.device)

            # construct batch graph
            mi_matrix_test = get_batched_mi_matrix(args, train_x)
            graph_batched = get_batched_graph(mi_matrix_test)

            # print('processing!')

            for i in range(len(graph_batched.nodes)):
                with torch.no_grad():
                    mask = torch.zeros_like(test_x[0])
                    mask[i] = 1
                    train_x_masked = train_x * mask
                    graph_batched.nodes[i]['x'] = torch.concatenate([
                        # torch.mean(train_x[i]).float(),
                        # torch.std(train_x[i]).float(),
                        source_model.get_feature(train_x_masked).detach(),
                        source_model.get_feature(train_x_masked).std(dim=0).detach(),
                        # torch.std(tot_train_x[i]).float(),
                    ])

                    # graph_batched.nodes[i]['x'] = [torch.mean(source_model.get_feature(test_x_masked),
                    #                                               dim=0).detach()]
            graph_input = from_networkx(graph_batched).to(args.device)

            vanilla_out = F.softmax(source_model(train_x), dim=-1).detach()
            gnn_out = gnn(graph_input).squeeze()
            estimated_y = F.softmax(vanilla_out + gnn_out, dim=-1)

            loss = F.cross_entropy(estimated_y, torch.argmax(train_y, dim=-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
        print(f'epoch {epoch} loss is {loss_total}')


    # for epoch in range(500):
    #     vanilla_out = F.softmax(source_model(train_x), dim=-1).detach()
    #     gnn_out = gnn(graph_train_input).squeeze()
    #     out = F.softmax(vanilla_out + gnn_out, dim=-1)
    #     loss = F.cross_entropy(out, torch.argmax(train_y, dim=-1))
    #     optimizer.zero_grad()
    #     loss.backward()
    #     if epoch % 100 == 0:
    #         print(f'orig_label is : {train_y[0]}')
    #         print(f'orig_pred is : {vanilla_out[0]}')
    #         print(f'gnn_pred is : {gnn_out}')
    #         print(
    #             f'bef acc is : {torch.mean((torch.argmax(vanilla_out.detach(), dim=-1) == torch.argmax(train_y, dim=-1)).float())}')
    #         print(
    #             f'aft acc is : {torch.mean((torch.argmax(vanilla_out.detach() + gnn_out, dim=-1) == torch.argmax(train_y, dim=-1)).float())}')
    #         print(loss)
    #     optimizer.step()

    return gnn, graph_test_input