import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class GraphNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, cat_cls_len, cont_len):
        super(GraphNet, self).__init__()
        self.cat_cls_len = cat_cls_len
        self.cont_len = cont_len
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = torch.nn.Linear(32, num_classes)

        # TODO: linear layer for each categorical feature
        self.embedding_layer = [
            torch.nn.Linear(max(self.cat_cls_len), num_features) for _ in self.cat_cls_len
        ]

    def forward(self, data):
        num_x, cat_x, edge_index = data.num_x, data.cat_x, data.edge_index

        # Apply the linear embedding layer to the categorical features

        x = [num_x]
        # cls_size = max(self.cat_cls_len)
        for i in range(len(self.cat_cls_len)):
            x.append(self.embedding_layer[i](cat_x[i]).unsqueeze(0))

        x = torch.concatenate(x, dim=0).float()

        # print(x)

        # Apply the GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # TODO: concat all features from nodes
        x = global_mean_pool(x, data.batch) # TODO: or concat predefined logits
        # concat use
        x = self.fc(x)
        x = F.softmax(x, dim=-1)
        x = x - x.mean(dim=-1, keepdim=True)

        return x

    def _apply(self, fn):
        super(GraphNet, self)._apply(fn)
        # Apply the linear embedding layer to the categorical features
        for i in self.embedding_layer:
            i._apply(fn)
        return self
