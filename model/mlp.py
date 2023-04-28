import torch
import torch.nn as nn



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, dropout=0.0):
        super().__init__()

        self.layers = nn.Sequential()
        in_dim = input_dim
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            # self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU(inplace=True))
            # self.layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.layers.append(torch.nn.Linear(in_dim, output_dim))
    
    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs