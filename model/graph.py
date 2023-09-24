import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F



class GraphNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, cat_cls_len, cont_len, type=1):
        super(GraphNet, self).__init__()
        self.cat_cls_len = cat_cls_len
        self.cont_len = cont_len
        self.conv1 = GCNConv(num_features, 4, bias=True)
        # self.conv2 = GCNConv(8, 4, bias=True)

        print(f"Hello~~~: {cat_cls_len}")
        print(f"Hello 2~~~: {cont_len}")

        self.fc = torch.nn.Linear((len(cat_cls_len) + cont_len) * 4 + 2 * num_classes, num_classes)
        self.type = type

        # TODO: linear layer for each categorical feature
        self.embedding_layer = [
            torch.nn.Linear(max(self.cat_cls_len), num_features) for _ in self.cat_cls_len
        ]

    def forward(self, data, vanilla_out):
        num_x, cat_x, edge_index, edge_weight = data.num_x, data.cat_x, data.edge_index, data.edge_weights

        # Apply the linear embedding layer to the categorical features

        x = [num_x]
        # cls_size = max(self.cat_cls_len)
        if self.type in [0, 1]:
            for i in range(len(self.cat_cls_len)):
                x.append(self.embedding_layer[i](cat_x[i]).unsqueeze(0))
        elif self.type in [2]:
            for i in range(len(self.cat_cls_len)):
                x.append(cat_x[i].unsqueeze(0))
        else:
            raise NotImplementedError

        x = torch.concatenate(x, dim=0).float()

        # print(x)

        # Apply the GCN layers
        x = self.conv1(x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        # x = self.conv2(x, edge_index=edge_index, edge_weight=edge_weight)
        x = x.reshape(1, -1).repeat(len(vanilla_out), 1)
        x = torch.cat([vanilla_out, x], dim=-1)

        print(f"x.shape: {x.shape}")

        # if self.type in [0, 1, 2]:
        #     x = global_mean_pool(x, data.batch)
        # # elif self.type in [2]:
        # #     pass
        # else:
        #     raise NotImplementedError
        # concat use
        x = self.fc(x)
        # x = F.softmax(x, dim=-1)
        # x = x - x.mean(dim=-1, keepdim=True)
        return x

    def _apply(self, fn):
        super(GraphNet, self)._apply(fn)
        # Apply the linear embedding layer to the categorical features
        for i in self.embedding_layer:
            i._apply(fn)
        return self

class GraphNet_tempscale(torch.nn.Module):
    def __init__(self, num_features, num_classes, cat_cls_len, cont_len, type=1):
        super(GraphNet_tempscale, self).__init__()
        self.cat_cls_len = cat_cls_len
        self.cont_len = cont_len
        self.conv1 = GCNConv(num_features, 16, bias=True)
        self.conv2 = GCNConv(16, 8, bias=True)

        self.fc = torch.nn.Linear((len(cat_cls_len) + cont_len) * 8 + 2 * num_classes, num_classes)
        self.type = type

        # TODO: linear layer for each categorical feature
        self.embedding_layer = [
            torch.nn.Linear(max(self.cat_cls_len), num_features) for _ in self.cat_cls_len
        ]

    def forward(self, data, vanilla_out, prob_dist=None):
        num_x, cat_x, edge_index, edge_weight = data.num_x, data.cat_x, data.edge_index, data.edge_weights

        x = [num_x]
        for i in range(len(self.cat_cls_len)):
            x.append(self.embedding_layer[i](cat_x[i]).unsqueeze(0))
        x = torch.concatenate(x, dim=0).float()


        # Apply the GCN layers
        x = self.conv1(x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = x.reshape(1, -1).repeat(len(vanilla_out), 1)
        x = torch.cat([vanilla_out, x, prob_dist.repeat(x.shape[0], 1)], dim=-1)

        x = self.fc(x)
        x = F.softplus(x, beta=1.1)
        # print(x)

        return x

    def _apply(self, fn):
        super(GraphNet_tempscale, self)._apply(fn)
        # Apply the linear embedding layer to the categorical features
        for i in self.embedding_layer:
            i._apply(fn)
        return self

