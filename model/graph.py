import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class GraphNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GraphNet, self).__init__()
        self.conv1 = GCNConv(num_features, 256)
        self.conv2 = GCNConv(256, 64)
        self.fc = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)

        # add pooling
        # x = global_mean_pool(x, data.batch)
        x = self.fc(x)

        return x
