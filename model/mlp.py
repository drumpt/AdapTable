import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, dropout=0.0):
        super().__init__()

        self.layers = nn.Sequential()
        self.cls_layer = nn.Sequential()
        in_dim = input_dim
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            # self.layers.append(nn.BatchNorm1d(hidden_dim))
            # self.layers.append(nn.LayerNorm(hidden_dim))
            # self.layers.append(nn.InstanceNorm1d(hidden_dim))
            # self.layers.append(nn.GroupNorm(int(np.sqrt(hidden_dim)), hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.cls_layer.append(nn.Linear(in_dim, output_dim))

    def forward(self, inputs):
        outputs = self.layers(inputs)
        outputs = self.cls_layer(outputs)
        return outputs

    def get_feature(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class MLP_MAE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, dropout=0.0):
        super().__init__()

        self.encoder = nn.Sequential()
        self.recon_head = nn.Sequential()
        # self.main_head = nn.Sequential()
        self.cls_main_head = nn.Sequential()

        in_dim = input_dim
        # self.encoder.append(nn.LayerNorm(in_dim))
        for _ in range(n_layers - 3):
            self.encoder.append(nn.Linear(in_dim, hidden_dim))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.encoder.append(nn.Linear(in_dim, hidden_dim))
        self.encoder.append(nn.ReLU())

        self.recon_head.append(nn.Linear(hidden_dim, 2 * output_dim))
        self.recon_head.append(nn.ReLU())
        self.recon_head.append(nn.Linear(2 * output_dim, output_dim))

        # self.bin_head.append(nn.ReLU())
        # self.bin_head.append(nn.Dropout(dropout))
        # self.bin_head.append(nn.Linear(hidden_dim, n_bin))

        # self.main_head.append(nn.Linear(hidden_dim, hidden_dim))
        # self.main_head.append(nn.ReLU())
        # self.main_head.append(nn.Dropout(dropout))
        # self.main_head.append(nn.Linear(hidden_dim, output_dim))

        self.cls_main_head.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, inputs):
        hidden_repr = self.encoder(inputs)
        recon_out = self.recon_head(hidden_repr)
        # main_out = self.main_head(hidden_repr)
        main_out = self.cls_main_head(hidden_repr)
        return recon_out, main_out

    def get_feature(self, inputs):
        hidden_repr = self.encoder(inputs)
        # recon_out = self.recon_head(hidden_repr)
        # main_out = self.main_head(hidden_repr)
        return hidden_repr

    def log_single_encoder_layer(self):
        print([param for param in self.encoder.parameters()][-1])


class MLP_Regression(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, n_bin, dropout=0.0):
        super().__init__()

        self.encoder = nn.Sequential()
        self.bin_head = nn.Sequential()
        # self.main_head = nn.Sequential()
        self.cls_main_head = nn.Sequential()

        in_dim = input_dim
        # self.encoder.append(nn.LayerNorm(in_dim))
        for _ in range(n_layers - 3):
            self.encoder.append(nn.Linear(in_dim, hidden_dim))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.encoder.append(nn.Linear(in_dim, hidden_dim))
        self.encoder.append(nn.ReLU())

        self.bin_head.append(nn.Linear(hidden_dim, 2 * n_bin))
        self.bin_head.append(nn.ReLU())
        self.bin_head.append(nn.Linear(2 * n_bin, n_bin))

        # self.bin_head.append(nn.ReLU())
        # self.bin_head.append(nn.Dropout(dropout))
        # self.bin_head.append(nn.Linear(hidden_dim, n_bin))

        # self.main_head.append(nn.Linear(hidden_dim, hidden_dim))
        # self.main_head.append(nn.ReLU())
        # self.main_head.append(nn.Dropout(dropout))
        # self.main_head.append(nn.Linear(hidden_dim, output_dim))

        self.cls_main_head.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, inputs):
        hidden_repr = self.encoder(inputs)
        bin_hidden_repr = hidden_repr
        # bin_hidden_repr = hidden_repr / torch.sum(hidden_repr, dim=1, keepdim=True)
        # bin_out = F.softmax(self.bin_head(bin_hidden_repr), dim=1)
        bin_out = self.bin_head(bin_hidden_repr)
        main_out = self.cls_main_head(hidden_repr)
        return bin_out, main_out

    def get_feature(self, inputs):
        hidden_repr = self.encoder(inputs)
        return hidden_repr

    def log_single_encoder_layer(self):
        print([param for param in self.encoder.parameters()][-1])


class Conv_MAE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, dropout=0.0):
        super().__init__()

        self.encoder = nn.Sequential()
        self.recon_head = nn.Sequential()
        self.main_head = nn.Sequential()
        self.cls_main_head = nn.Sequential()

        in_dim = input_dim
        # self.encoder.append(nn.LayerNorm(in_dim))
        for _ in range(n_layers - 3):
            # self.encoder.append(nn.Linear(in_dim, hidden_dim))
            self.encoder.append(nn.Conv1d(in_dim, hidden_dim, kernel_size=3, stride=1, padding=1))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.encoder.append(nn.Linear(in_dim, hidden_dim))

        self.recon_head.append(nn.Linear(hidden_dim, input_dim))

        self.main_head.append(nn.Linear(hidden_dim, hidden_dim))
        self.main_head.append(nn.ReLU())
        self.main_head.append(nn.Dropout(dropout))
        # self.main_head.append(nn.Linear(hidden_dim, output_dim))

        self.cls_main_head.append(nn.Linear(hidden_dim, output_dim))

        # add softmax at end
        # self.cls_main_head.append(nn.Softmax(dim=1))

    def forward(self, inputs):
        hidden_repr = self.encoder(inputs)
        recon_out = self.recon_head(hidden_repr)
        main_out = self.main_head(hidden_repr)
        main_out = self.cls_main_head(main_out)
        return recon_out, main_out

    def get_feature(self, inputs):
        hidden_repr = self.encoder(inputs)
        # recon_out = self.recon_head(hidden_repr)
        # main_out = self.main_head(hidden_repr)
        return hidden_repr

    def log_single_encoder_layer(self):
        print([param for param in self.encoder.parameters()][-1])