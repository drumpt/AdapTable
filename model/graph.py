import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F



class GraphNet(torch.nn.Module):
    def __init__(self, args, num_features, num_classes, cat_cls_len, cont_len):
        super().__init__()
        self.args = args
        self.cat_cls_len = cat_cls_len
        self.cont_len = cont_len
        self.num_features = num_features
        self.num_classes = num_classes

        self.conv = GCNConv(num_features, 128, bias=False)
        self.fc = torch.nn.Linear(128 + num_classes, 1, bias=True)

        # TODO: linear layer for each categorical feature
        self.embedding_layer = [
            torch.nn.Linear(cat_len, 1, bias=False) for cat_len in self.cat_cls_len
        ]


    def forward(self, data, vanilla_out):
        num_x, cat_x, edge_index, edge_weight = data.num_x, data.cat_x, data.edge_index, data.edge_weights
    
        print(f"{num_x.shape=}")
        print(f"{cat_x.shape=}")
        print(f"{edge_index=}")
        print(f"{edge_weight=}")

        x = [torch.tensor(num_x).to(self.args.device)]
        for idx, cat_len in enumerate(self.cat_cls_len):
            x.append(self.embedding_layer[idx](cat_x[idx][:, :cat_len]).squeeze().unsqueeze(0))


        x = torch.cat(x, dim=0).float()
        print(f"input feature {x.shape=}")

        if x.shape[1] != self.num_features:
            # pad with zeros to match the number of features
            x = torch.cat([x, torch.zeros(x.shape[0], self.num_features - x.shape[1]).to(x.device)], dim=-1)
        print(f"input feature {x.shape=}")

        # print(f"{x.shape=}")
        # print(f"{edge_index=}")
        # print(f"{edge_weight=}")

        # conv layer
        x = self.conv(x, edge_index=edge_index, edge_weight=edge_weight)
        print(f"after conv {x.shape=}")
        x = F.relu(x)

        # pooling layer
        x = global_mean_pool(x, data.batch)
        print(f"after global_mean_pool {x.shape=}")

        # match vanilla_out and append
        x = x.repeat(len(vanilla_out), 1)
        print(f"repeat {x.shape=}")
        x = torch.cat([vanilla_out, x], dim=-1)
        print(f"concat {x.shape=}")

        # fc layer
        x = self.fc(x)
        print(f"fc output {x.shape=}")

        # turn into temperature
        t = F.softplus(x, beta=1.1)
        print(f"softplus {t.shape=}")
        t = t.repeat(1, self.num_classes)
        print(f"rep[eat] {t.shape=}")

        x = torch.div(vanilla_out, t)

        print(f"final out {x.shape=}")
        return x


    def _apply(self, fn):
        super()._apply(fn)
        # Apply the linear embedding layer to the categorical features
        for i in self.embedding_layer:
            i._apply(fn)
        return self