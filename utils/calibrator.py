import torch.optim

from model.graph import *
from utils.calibration_loss_fn import *
from utils.utils import *



class Calibrator():
    def __init__(self, args, dataset, source_model):
        # print("2-1-1")
        self.args = args
        self.dataset = dataset

        self.model = source_model
        self.model.requires_grad_(False)
        # print("2-1-2")

        # parameters
        self.gnn_epochs = self.args.posttrain_epochs
        self.shrinkage_factor = self.args.posttrain_shrinkage_factor
        self.lr = self.args.posttrain_lr
        # print("2-1-3")

        # graph data
        from data.graph_data import GraphDataset
        self.train_graph_dataset = GraphDataset(args, dataset)
        # print("2-1-4")

        self.gnn = GraphNet(
            args=args,
            cat_cls_len=self.train_graph_dataset.cat_len_per_node,
            cont_len=len(self.train_graph_dataset.cont_indices),
            num_features=self.args.test_batch_size,
            num_classes=dataset.train_y.shape[1],
        ).to(args.device).float()
        # print("2-1-5")

        self.gnn.requires_grad_(True)
        # self.gnn.train()
        self.model.requires_grad_(False)
        # print("2-1-6")

    def train_gnn(self):
        print('========================GNN Training Start========================')
        prob_dist = torch.mean(torch.tensor(self.dataset.train_y), dim=0).to(self.args.device).float()
        loss_fn = Posttrain_loss(self.shrinkage_factor, prob_dist)

        train_graph_dataset = self.train_graph_dataset
        train_len = len(train_graph_dataset.created_batches)
        valid_len = len(self.dataset.posttrain_valid_loader)

        optimizer = torch.optim.Adam(
            self.gnn.parameters(),
            lr=self.lr,
            weight_decay=5e-3,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_len, eta_min=0)

        self.model.requires_grad_(False)

        best_valid_loss = np.inf
        best_model = None
        best_epoch = 0
        patience = self.args.posttrain_patience

        num_iter = 0

        for epoch in range(self.gnn_epochs):
            loss_total = 0

            bef_adapt_list = []
            aft_adapt_list = []
            vanilla_out_list = []
            calibrated_out_list = []
            label_list = []

            # optimizer.zero_grad()
            for batched_train_x, batched_graph, batched_train_y in zip(train_graph_dataset.created_batches, train_graph_dataset.created_graph_batches, train_graph_dataset.created_batches_cls):
                batched_train_x, batched_train_y = batched_train_x.to(self.args.device).float(), batched_train_y.to(
                    self.args.device).float()
                batched_graph = batched_graph.to(self.args.device)

                num_iter += 1

                with torch.no_grad():
                    vanilla_out = self.model(batched_train_x).detach()


                prob_dist = torch.sum(batched_train_y, dim=0) / len(batched_train_y)
                estimated_y = self.gnn(batched_graph, vanilla_out).squeeze()

                loss = loss_fn(estimated_y, batched_train_y)
                loss_total += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # log batch accuaracy
                bef_adapt = (torch.argmax(vanilla_out, dim=-1) == torch.argmax(batched_train_y, dim=-1)).sum().item() / len(batched_train_y)
                aft_adapt = (torch.argmax(estimated_y, dim=-1) == torch.argmax(batched_train_y, dim=-1)).sum().item() / len(batched_train_y)

                bef_adapt_list.append(bef_adapt)
                aft_adapt_list.append(aft_adapt)

                vanilla_out_list.extend(vanilla_out.detach().cpu().tolist())
                calibrated_out_list.extend(estimated_y.detach().cpu().tolist())
                label_list.extend(torch.argmax(batched_train_y, dim=-1).detach().cpu().tolist())

            with torch.no_grad():
                valid_loss_list = []
                for valid_x, valid_y in self.dataset.posttrain_valid_loader:
                    valid_x, valid_y = valid_x.to(self.args.device), valid_y.to(self.args.device)
                    calibrated_y = self.get_gnn_out(self.model, valid_x).detach()
                    valid_loss = loss_fn(calibrated_y.to(self.args.device), valid_y)
                    valid_loss_list.append(valid_loss.item())

                valid_loss = np.sum(valid_loss_list)

                if valid_loss < best_valid_loss:
                    patience = self.args.posttrain_patience
                    best_epoch = epoch
                    best_valid_loss = valid_loss
                    best_model = deepcopy(self.gnn)
                else:
                    patience -= 1

            if patience == 0:
                break

            print(f"posttrain epoch {epoch} | train_loss {loss_total / train_len:.4f}, valid_loss {valid_loss / valid_len:.4f}")
        best_model.requires_grad_(False)
        best_model.eval()
        print('best model saved at epoch ', best_epoch)
        print('========================GNN Training Done========================')
        return best_model

    def get_gnn_out(self, model, batch, wo_softmax=False):
        from data.graph_data import GraphDataset
        self.gnn = self.gnn.eval()
        test_graph_data = GraphDataset.create_test_graph(self.args, self.dataset, batch)

        vanilla_out = model(batch).detach()
        estimated_y = self.gnn(test_graph_data, vanilla_out)

        if wo_softmax:
            return estimated_y
        else:
            estimated_y = F.softmax(estimated_y, dim=-1)
            return estimated_y