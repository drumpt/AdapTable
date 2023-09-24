import numpy as np

from utils.graph_embedding import get_mi_matrix, get_node2vec_embedding, get_nx_graph
import torch
from torch_geometric.utils import from_networkx
from model.graph import *
from utils.calibration_loss_fn import *
from utils.graph_embedding import get_batched_mi_matrix, get_batched_graph
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
from utils.utils import *




# class RowwiseGraphNet():
#     def __init__(self, args, dataset, source_model):
#         self.gnn_epochs = 20
#         self.args = args
#         self.dataset = dataset

#         # construct gnn
#         # self.gnn = GraphNet(
#         #     num_features=dataset.train_x.shape[1],
#         #     num_classes=dataset.train_y.shape[1]
#         # ).to(args.device)

#         from data.graph_data import GraphDataset
#         self.train_graph_dataset = GraphDataset(args, dataset)

#         # construct gnn
#         self.gnn = GraphNet(
#             num_features=2,
#             num_classes=dataset.train_y.shape[1],
#             cat_cls_len=self.train_graph_dataset.cat_len_per_node,
#             cont_len=len(self.train_graph_dataset.cont_indices)
#         ).to(args.device).float()

#         self.model = source_model
#         self.model.requires_grad_(False)

#     def train_gnn(self):
#         train_x, train_y = torch.tensor(self.dataset.train_x).float().to(self.args.device), \
#                            torch.tensor(self.dataset.train_y).float().to(self.args.device)
#         valid_x, valid_y = torch.tensor(self.dataset.valid_x).float().to(self.args.device), \
#                            torch.tensor(self.dataset.valid_y).float().to(self.args.device)
#         optimizer = torch.optim.AdamW(self.gnn.parameters(), lr=0.001)

#         valid_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
#                                                        batch_size=self.args.test_batch_size, shuffle=True, drop_last=True)

#         best_gnn = deepcopy(self.gnn)
#         best_valid_loss = np.inf

#         for epoch in range(self.gnn_epochs):
#             loss_total = 0
#             train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
#                                                            batch_size=self.args.test_batch_size, shuffle=True,
#                                                            drop_last=True)
#             for batched_train_x, batchced_train_y in tqdm(train_dataloader):
#                 batched_train_x, batchced_train_y = batched_train_x.to(self.args.device), batchced_train_y.to(self.args.device)

#                 # construct adjacency matrxi via cossim between row's features
#                 with torch.no_grad():
#                     batched_feat_x = self.model.get_feature(batched_train_x)
#                     normalized_batched_feat_x = F.normalize(batched_feat_x, dim=-1)
#                     adj_matrix = torch.matmul(normalized_batched_feat_x, normalized_batched_feat_x.T)

#                 # construct graph
#                 batched_graph = from_networkx(get_batched_graph(adj_matrix))

#                 # fill in x values - the actual input features
#                 batched_graph.x = batched_train_x

#                 # cast to device
#                 batched_graph = batched_graph.to(self.args.device)

#                 vanilla_out = self.model(batched_train_x).detach()
#                 gnn_out = self.gnn(batched_graph)
#                 estimated_y = F.softmax(vanilla_out + gnn_out, dim=-1)

#                 loss = F.cross_entropy(estimated_y, torch.argmax(batchced_train_y, dim=-1))
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 loss_total += loss.item()
#             print(f'epoch {epoch} loss is {loss_total / len(train_dataloader)}')

#             if epoch % 1 == 0:
#                 valid_loss = 0
#                 with torch.no_grad():
#                     for valid_x, valid_y in tqdm(valid_dataloader):
#                         # log loss on validation set
#                         vanilla_out = self.model(valid_x).detach()

#                         batched_feat_x = self.model.get_feature(valid_x)
#                         normalized_batched_feat_x = F.normalize(batched_feat_x, dim=-1)
#                         adj_matrix = torch.matmul(normalized_batched_feat_x, normalized_batched_feat_x.T)

#                         # construct graph
#                         batched_graph = from_networkx(get_batched_graph(adj_matrix))

#                         # fill in x values - the actual input features
#                         batched_graph.x = valid_x

#                         # cast to device
#                         batched_graph = batched_graph.to(self.args.device)

#                         gnn_out = self.gnn(batched_graph)

#                         estimated_y = F.softmax(vanilla_out + gnn_out, dim=-1)
#                         valid_loss += F.cross_entropy(estimated_y, torch.argmax(valid_y, dim=-1))

#                     if valid_loss < best_valid_loss:
#                         best_valid_loss = valid_loss
#                         best_gnn = deepcopy(self.gnn)

#                     print(f'epoch {epoch} validation loss is {valid_loss.item() / len(valid_dataloader)}')
#         self.gnn = best_gnn
#         return self.gnn

#     def get_gnn_out(self, data):
#         # construct adj matrix via features
#         with torch.no_grad():
#             feat_x = self.model.get_feature(data)
#             normalized_feat_x = F.normalize(feat_x, dim=-1)
#             adj_matrix = torch.matmul(normalized_feat_x, normalized_feat_x.T)

#         # construct graph
#         graph = from_networkx(get_batched_graph(adj_matrix))
#         graph.x = data
#         graph = graph.to(self.args.device)

#         # get gnn out
#         gnn_out = self.gnn(graph)

#         # get vanilla out
#         vanilla_out = F.softmax(self.model(data), dim=-1).detach()

#         # get estimated out
#         estimated_out = F.softmax(vanilla_out + gnn_out, dim=-1)

#         return estimated_out



class ColumnwiseGraphNet():
    def __init__(self, args, dataset, soource_model):
        self.args = args
        self.dataset = dataset

        self.model = soource_model
        self.model.requires_grad_(False)

        # graph data
        from data.graph_data import GraphDataset
        self.train_graph_dataset = GraphDataset(args, dataset)

        # construct gnn
        self.gnn = GraphNet(
            num_features=2,
            num_classes=dataset.train_y.shape[1],
            cat_cls_len=self.train_graph_dataset.cat_len_per_node,
            cont_len=len(self.train_graph_dataset.cont_indices)
        ).to(args.device).float()
        self.gnn.requires_grad_(True)
        self.gnn.train()

        # parameters
        self.gnn_epochs = 20
        self.shrinkage_factor = 1
        self.lr = 0.05

    def train_gnn(self):
        train_graph_dataset = self.train_graph_dataset
        optimizer = torch.optim.AdamW(self.gnn.parameters(), lr=self.lr)

        for epoch in range(self.gnn_epochs):
            loss_total = 0
            optimizer.zero_grad()
            for batched_train_x, batched_graph, batched_train_y in tqdm(
                    zip(train_graph_dataset.created_batches, train_graph_dataset.created_graph_batches,
                        train_graph_dataset.created_batches_cls)):
                batched_train_x, batched_train_y = batched_train_x.to(self.args.device).float(), batched_train_y.to(
                    self.args.device).float()
                batched_graph = batched_graph.to(self.args.device)

                vanilla_out = self.model(batched_train_x).detach()  # currently vanilla outs are logits
                gnn_out = self.gnn(batched_graph, vanilla_out) * self.shrinkage_factor  # currently gnn outs are logits
                # estimated_y = F.softmax(vanilla_out, dim=-1) + gnn_out
                estimated_y = vanilla_out + gnn_out

                # print(f"vanilla_out: {vanilla_out[0]}")
                # print(f"gnn_out: {gnn_out}")
                # print(f"estimated_y: {estimated_y[0]}")
                # print(f"torch.argmax(batched_train_y, dim=-1)[0]: {torch.argmax(batched_train_y, dim=-1)[0]}")

                # print(f"F.softmax(vanilla_out, dim=-1): {F.softmax(vanilla_out, dim=-1)[0]}")
                # print(f"F.softmax(vanilla_out, dim=-1): {F.softmax(vanilla_out, dim=-1)[0]}")
                # print(f"gnn_out: {gnn_out[0]}")
                # print(f"estimated_y: {estimated_y}")

                loss = F.cross_entropy(estimated_y, torch.argmax(batched_train_y, dim=-1))
                loss += F.l1_loss(gnn_out, torch.zeros_like(gnn_out)) * 0.001  # regularization term

                loss /= len(train_graph_dataset.created_batches)
                loss_total += loss.item()

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                with torch.no_grad():
                    # logging
                    y_cnt = np.unique(torch.argmax(batched_train_y, dim=1).cpu().numpy(), return_counts=True)
                    # print(f'batch distribution is : [{y_cnt}]]')
                    # print(f'bias is : [{gnn_out}]')

                    vanilla_acc = (torch.argmax(vanilla_out, dim=-1) == torch.argmax(batched_train_y, dim=-1)).sum()
                    calibrated_acc = (torch.argmax(F.softmax(vanilla_out, dim=-1) + gnn_out, dim=-1) == torch.argmax(batched_train_y, dim=-1)).sum()

                    # print(f'vanilla acc is : {vanilla_acc / len(batched_train_y)}')
                    # print(f'calibrated acc is : {calibrated_acc / len(batched_train_y)}')
                    # print('')

            print(f'epoch {epoch} loss is {loss_total / len(train_graph_dataset.created_batches)}')

        self.model.requires_grad_(True)
        self.gnn.requires_grad_(False)
        return self.gnn

    def get_gnn_out(self, batch, vanilla_out):
        from data.graph_data import GraphDataset
        test_graph_data = GraphDataset.create_test_graph(self.args, self.dataset, batch)
        gnn_out = self.gnn(test_graph_data, vanilla_out) * self.shrinkage_factor
        source_out = self.model(batch).detach()
        # estimated_out = F.softmax(source_out, dim=-1) + gnn_out
        estimated_out = source_out + gnn_out
        return estimated_out


class ColumnwiseGraphNet_tempscaling():
    def __init__(self, args, dataset, source_model):
        self.args = args
        self.dataset = dataset

        self.model = source_model
        self.model.requires_grad_(False)
        self.type = 1

        # parameters
        self.gnn_epochs = self.args.posttrain_epochs
        self.shrinkage_factor = 0.5
        self.lr = self.args.posttrain_lr
        self.num_batches = 50

        # graph data
        from data.graph_data import GraphDataset
        self.train_graph_dataset = GraphDataset(args, dataset, type=self.type, num_batches=self.num_batches)

        self.gnn = GraphNet_tempscale(
            num_features=2,
            num_classes=dataset.train_y.shape[1],
            cat_cls_len=self.train_graph_dataset.cat_len_per_node,
            cont_len=len(self.train_graph_dataset.cont_indices),
            type=self.type,
        ).to(args.device).float()


        self.gnn.requires_grad_(True)
        self.gnn.train()
        self.model.requires_grad_(False)

    def train_gnn(self):
        # reg_loss_fn = CAGCN_loss()
        loss_fn = FocalLoss(gamma=10)
        reg_loss_fn = CAGCN_loss()

        train_graph_dataset = self.train_graph_dataset
        optimizer = torch.optim.AdamW(self.gnn.parameters(), lr=self.lr)

        best_ece_score = np.inf
        best_model = None
        best_epoch = 0

        for epoch in range(self.gnn_epochs):
            loss_total = 0

            bef_adapt_list = []
            aft_adapt_list = []
            vanilla_out_list = []
            calibrated_out_list = []
            label_list = []

            for batched_train_x, batched_graph, batched_train_y in tqdm(
                    zip(train_graph_dataset.created_batches, train_graph_dataset.created_graph_batches,
                        train_graph_dataset.created_batches_cls)):
                batched_train_x, batched_train_y = batched_train_x.to(self.args.device).float(), batched_train_y.to(
                    self.args.device).float()
                batched_graph = batched_graph.to(self.args.device)

                with torch.no_grad():
                    vanilla_out = self.model(batched_train_x).detach()
                    # [0.1 0.2]

                # prob distribution of y
                prob_dist = torch.sum(batched_train_y, dim=0) / len(batched_train_y)

                gnn_out = self.gnn(batched_graph, vanilla_out, prob_dist).squeeze()
                gnn_out = gnn_out.repeat(vanilla_out.size(1), 1).transpose(0, 1)

                estimated_y = torch.mul(vanilla_out, gnn_out)

                # loss = loss_fn(estimated_y, batched_train_y)

                estimated_y = F.softmax(estimated_y, dim=-1)
                vanilla_out = F.softmax(vanilla_out, dim=-1)

                # loss += 0.5 * reg_loss_fn(estimated_y, batched_train_y)

                loss = reg_loss_fn(estimated_y, batched_train_y)

                loss_total += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # log batch accuaracy
                bef_adapt = (torch.argmax(vanilla_out, dim=-1) == torch.argmax(batched_train_y, dim=-1)).sum().item() / len(batched_train_y)
                aft_adapt = (torch.argmax(estimated_y, dim=-1) == torch.argmax(batched_train_y, dim=-1)).sum().item() / len(batched_train_y)

                bef_adapt_list.append(bef_adapt)
                aft_adapt_list.append(aft_adapt)

                vanilla_out_list.extend(vanilla_out.detach().cpu().tolist())
                calibrated_out_list.extend(estimated_y.detach().cpu().tolist())
                label_list.extend(torch.argmax(batched_train_y, dim=-1).detach().cpu().tolist())

            with torch.no_grad():
                ece_loss_fn = ECELoss()
                ece_loss_list = []
                for valid_x, valid_y in self.dataset.valid_loader:
                    valid_x, valid_y = valid_x.to(self.args.device), valid_y.to(self.args.device)
                    estimated_y = self.model(valid_x).detach().cpu()
                    prob_dist = torch.sum(valid_y, dim=0) / len(valid_y)
                    calibrated_y = self.get_gnn_out(self.model, valid_x, prob_dist).detach().cpu()
                    ece_loss = ece_loss_fn(calibrated_y.to(self.args.device), torch.argmax(valid_y, dim=-1))
                    ece_loss_list.append(ece_loss.item())
                ece_loss = np.sum(ece_loss_list)

                if ece_loss < best_ece_score:
                    best_epoch = epoch + 1
                    # best_ece_score = ece_loss
                    best_model = deepcopy(self.gnn)


            print(f'epoch {epoch} loss is {loss_total / len(train_graph_dataset.created_batches)}')
            print(f'epoch {epoch} bef_adapt is {np.mean(bef_adapt_list)}')
            print(f'epoch {epoch} aft_adapt is {np.mean(aft_adapt_list)}')
            print(f'epoch {epoch} ece is {ece_loss / len(self.dataset.valid_loader)}')

        best_model.requires_grad_(False)
        best_model.eval()
        self.model.requires_grad_(True)
        print('best model saved at epoch ', best_epoch)
        return best_model

    def get_gnn_out(self, model, batch, prob_dist):
        from data.graph_data import GraphDataset
        test_graph_data = GraphDataset.create_test_graph(self.args, self.dataset, batch)

        vanilla_out = model(batch)
        gnn_out = self.gnn(test_graph_data, vanilla_out, prob_dist)
        estimated_y = torch.mul(vanilla_out, gnn_out.detach())
        estimated_y = F.softmax(estimated_y, dim=-1)
        return estimated_y


# class ColumnwiseGraphNet_rowfeat():
#     def __init__(self, args, dataset, soource_model):
#         self.args = args
#         self.dataset = dataset

#         self.model = soource_model
#         self.model.requires_grad_(False)

#         # graph data
#         from data.graph_data import GraphDataset
#         self.train_graph_dataset = GraphDataset(args, dataset, type=2)

#         # construct gnn
#         self.gnn = GraphNet(
#             num_features=args.test_batch_size,
#             num_classes=dataset.train_y.shape[1],
#             cat_cls_len=self.train_graph_dataset.cat_len_per_node,
#             cont_len=len(self.train_graph_dataset.cont_indices),
#             type=2
#         ).to(args.device).float()
#         self.gnn.requires_grad_(True)
#         self.gnn.train()

#         print(f"self.train_graph_dataset.cat_len_per_node: {self.train_graph_dataset.cat_len_per_node}")
#         print(f"len(self.train_graph_dataset.cont_indices): {len(self.train_graph_dataset.cont_indices)}")

#         # parameters
#         self.gnn_epochs = 50
#         self.shrinkage_factor = 1
#         self.lr = 0.0001

#     def train_gnn(self):
#         train_graph_dataset = self.train_graph_dataset
#         optimizer = torch.optim.AdamW(self.gnn.parameters(), lr=self.lr)
#         self.model.requires_grad_(False)

#         split = int(len(train_graph_dataset.created_batches) * 0.8)

#         best_valid_loss = float('inf')

#         for epoch in range(self.gnn_epochs):
#             loss_total = 0
#             train_loader = zip(
#                 train_graph_dataset.created_batches[:split],
#                 train_graph_dataset.created_graph_batches[:split],
#                 train_graph_dataset.created_batches_cls[:split]
#             )
#             valid_loader = zip(
#                 train_graph_dataset.created_batches[split:],
#                 train_graph_dataset.created_graph_batches[split:],
#                 train_graph_dataset.created_batches_cls[split:]
#             )

#             # print(f"train_graph_dataset.created_batches[:split]: {train_graph_dataset.created_batches[:split]}")
#             # print(f"train_graph_dataset.created_batches[:split]: {train_graph_dataset.created_batches_cls[:split]}")

#             # print(f"train_graph_dataset.created_batches[:split]: {train_graph_dataset.created_batches[split:]}")
#             # print(f"train_graph_dataset.created_batches[:split]: {train_graph_dataset.created_batches_cls[split:]}")

#             for batched_train_x, batched_graph, batched_train_y in tqdm(train_loader):
#                 print(f"batched_train_x: {batched_train_x.shape}")
#                 print(f"batched_graph: {batched_graph}")

#                 if len(batched_train_x) != self.args.test_batch_size:
#                     continue

#                 batched_train_x, batched_train_y = batched_train_x.to(self.args.device).float(), batched_train_y.to(
#                     self.args.device).float()
#                 batched_graph = batched_graph.to(self.args.device)

#                 vanilla_out = self.model(batched_train_x).detach()  # currently vanilla outs are logits
#                 print(f"vanilla_out: {vanilla_out.shape}")
#                 print(f"batched_graph")
#                 gnn_out = self.gnn(batched_graph) * self.shrinkage_factor  # currently gnn outs are logits
#                 estimated_y = F.softmax(vanilla_out, dim=-1) + gnn_out

#                 loss = F.cross_entropy(estimated_y, torch.argmax(batched_train_y, dim=-1))
#                 loss_total += loss.item()

#                 optimizer.zero_grad()
#                 loss.backward(retain_graph=True)
#                 optimizer.step()
#                 # with torch.no_grad():
#                 #     # logging
#                 #     y_cnt = np.unique(torch.argmax(batched_train_y, dim=1).cpu().numpy(), return_counts=True)
#                 #     print(f'batch distribution is : [{y_cnt}]]')
#                 #     print(f'bias is : [{gnn_out[0]}]')
#                 #
#                 #     vanilla_acc = (torch.argmax(vanilla_out, dim=-1) == torch.argmax(batched_train_y, dim=-1)).sum()
#                 #     calibrated_acc = (torch.argmax(F.softmax(vanilla_out, dim=-1) + gnn_out, dim=-1) == torch.argmax(
#                 #         batched_train_y, dim=-1)).sum()
#                 #
#                 #     print(f'vanilla acc is : {vanilla_acc / len(batched_train_y)}')
#                 #     print(f'calibrated acc is : {calibrated_acc / len(batched_train_y)}')
#                 #     print('')

#                 # validation

#             print(f'epoch {epoch} loss is {loss_total}')

#             with torch.no_grad():
#                 valid_loss = 0
#                 for batched_valid_x, batched_graph, batched_valid_y in tqdm(valid_loader):
#                     vanilla_out = self.model(batched_valid_x).detach()  # currently vanilla outs are logits
#                     gnn_out = self.gnn(batched_graph) * self.shrinkage_factor  # currently gnn outs are logits
#                     estimated_y = F.softmax(vanilla_out, dim=-1) + gnn_out

#                     valid_loss += F.cross_entropy(estimated_y, torch.argmax(batched_valid_y, dim=-1)).item()
#                 if valid_loss < best_valid_loss:
#                     # print(f'new best valid loss! {valid_loss}')
#                     best_gnn = deepcopy(self.gnn)

#         self.model.requires_grad_(True)

#         self.gnn = best_gnn
#         self.gnn.requires_grad_(False)
#         return self.gnn

#     def get_gnn_out(self, batch):
#         with torch.no_grad():
#             from data.graph_data import GraphDataset
#             test_graph_data = GraphDataset.create_test_graph(self.args, self.dataset, batch, type=2)
#             # get gnn out
#             gnn_out = self.gnn(test_graph_data) * self.shrinkage_factor
#             print(f'gnn out is : {gnn_out[0]}')
#             source_out = self.model(batch).detach()

#             estimated_out = F.softmax(source_out, dim=-1) + gnn_out
#         return estimated_out




# def get_pretrained_graphnet_columnwise(args, dataset, source_model):
#     train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
#         args.device)
#     source_model.requires_grad_(False)
#
#     tot_train_x = train_x
#     test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
#         args.device)
#
#     from data.graph_data import GraphDataset
#     train_graph_dataset = GraphDataset(args, dataset)
#
#     # construct gnn
#     gnn = GraphNet(
#         num_features=2,
#         num_classes=dataset.train_y.shape[1],
#         cat_cls_len=train_graph_dataset.cat_len_per_node,
#         cont_len=len(train_graph_dataset.cont_indices)
#     ).to(args.device).float()
#     gnn.requires_grad_(True)
#     gnn.train()
#
#     optimizer = torch.optim.AdamW(gnn.parameters(), lr=0.005)
#     shrinkage_factor = 0.1
#     for epoch in range(50):
#         loss_total = 0
#         optimizer.zero_grad()
#         for batched_train_x, batched_graph, batched_train_y in tqdm(zip(train_graph_dataset.created_batches, train_graph_dataset.created_graph_batches, train_graph_dataset.created_batches_cls)):
#
#             batched_train_x, batched_train_y = batched_train_x.to(args.device).float(), batched_train_y.to(args.device).float()
#             batched_graph = batched_graph.to(args.device)
#
#             vanilla_out = source_model(batched_train_x).detach() # currently vanilla outs are logits
#             gnn_out = gnn(batched_graph) * shrinkage_factor # currently gnn outs are logits
#             estimated_y = F.softmax(vanilla_out, dim=-1) + gnn_out
#
#             loss = F.cross_entropy(estimated_y, torch.argmax(batched_train_y, dim=-1))
#             loss += F.l1_loss(gnn_out, torch.zeros_like(gnn_out)) * 0.1 # regularization term
#
#             loss /= len(train_graph_dataset.created_batches)
#             loss_total += loss.item()
#             loss.backward(retain_graph=True)
#
#         optimizer.step()
#         print(f'epoch {epoch} loss is {loss_total / len(train_graph_dataset.created_batches)}')
#
#     source_model.requires_grad_(True)
#     gnn.requires_grad_(False)
#     return deepcopy(gnn)



# def get_pretrained_gnn_tosource(args, dataset, batch, source_model, gnn):
#     train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
#         args.device)
#     source_model.requires_grad_(False)
#
#     from data.graph_data import GraphDataset
#     train_graph_dataset = GraphDataset(args, dataset)
#
#     # construct gnn
#     gnn = GraphNet(
#         num_features=2,
#         num_classes=dataset.train_y.shape[1],
#         cat_cls_len=train_graph_dataset.cat_len_per_node,
#         cont_len=len(train_graph_dataset.cont_indices)
#     ).to(args.device).float()
#     gnn.requires_grad_(True)
#     gnn.train()
#     pass