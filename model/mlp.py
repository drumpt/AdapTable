import numpy as np
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
            # self.layers.append(nn.LayerNorm(hidden_dim))
            # self.layers.append(nn.GroupNorm(int(np.sqrt(hidden_dim)), hidden_dim))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.layers.append(nn.Linear(in_dim, output_dim))
    
    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs



class MLP_MAE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, dropout=0.0):
        super().__init__()

        self.encoder = nn.Sequential()
        # self.recon_head = nn.Sequential()
        self.recon_mean = nn.Sequential()
        self.recon_std = nn.Sequential()
        self.main_head = nn.Sequential()

        in_dim = input_dim
        for _ in range(n_layers - 3):
            self.encoder.append(nn.Linear(in_dim, hidden_dim))
            self.encoder.append(nn.ReLU(inplace=True))
            self.encoder.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.encoder.append(nn.Linear(in_dim, hidden_dim))

        # self.recon_head.append(nn.Linear(in_dim, hidden_dim))
        # self.recon_head.append(nn.ReLU(inplace=True))
        # self.recon_head.append(nn.Dropout(dropout))
        # self.recon_head.append(nn.Linear(in_dim, input_dim))

        self.recon_mean.append(nn.Linear(in_dim, input_dim))
        self.recon_std.append(nn.Linear(in_dim, input_dim))

        self.main_head.append(nn.Linear(in_dim, hidden_dim))
        self.main_head.append(nn.ReLU(inplace=True))
        self.main_head.append(nn.Dropout(dropout))
        self.main_head.append(nn.Linear(in_dim, output_dim))


    def forward(self, inputs):
        hidden_repr = self.encoder(inputs)
        recon_mean = self.recon_mean(hidden_repr)
        recon_std = torch.exp(self.recon_std(hidden_repr))
        main_out = self.main_head(hidden_repr)
        return [recon_mean, recon_std], main_out