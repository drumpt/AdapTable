import numpy as np
import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer
from tab_transformer_pytorch import FTTransformer


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, dropout=0.0):
        super().__init__()

        self.layers = nn.Sequential()
        self.cls_layer = nn.Sequential()
        in_dim = input_dim
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
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
        self.main_head = nn.Sequential()
        self.cls_main_head = nn.Sequential()

        in_dim = input_dim
        # self.encoder.append(nn.LayerNorm(in_dim))
        for _ in range(n_layers - 3):
            self.encoder.append(nn.Linear(in_dim, hidden_dim))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.encoder.append(nn.Linear(in_dim, hidden_dim))

        self.recon_head.append(nn.Linear(hidden_dim, input_dim))

        self.main_head.append(nn.Linear(hidden_dim, hidden_dim))
        self.main_head.append(nn.ReLU())
        self.main_head.append(nn.Dropout(dropout))
        self.cls_head.append(nn.Linear(hidden_dim, output_dim))


    def forward(self, inputs):
        hidden_repr = self.encoder(inputs)
        recon_out = self.recon_head(hidden_repr)
        main_out = self.main_head(hidden_repr)
        main_out = self.cls_head(main_out)
        return recon_out, main_out


    def get_feature(self, inputs):
        hidden_repr = self.encoder(inputs)
        recon_out = self.recon_head(hidden_repr)
        main_out = self.main_head(hidden_repr)
        return main_out



# class TabNet(nn.Module): # TODO: (WIP) finish implementing this!
#     def __init__(self, args, dataset):
#         super().__init__()

#         self.model = TabNetRegressor(
#             device_name=args.device,
#             optimizer_fn=getattr(torch.optim, 'AdamW'),
#             optimizer_params=dict(lr=args.train_lr),
#         ) if dataset.out_dim == 1 else TabNetClassifier(
#             device_name=args.device,
#             optimizer_fn=getattr(torch.optim, 'AdamW'),
#             optimizer_params=dict(lr=args.train_lr),
#         )
#         self.recon_head = nn.Sequential()
#         self.recon_head.append(nn.Linear(256, dataset.in_dim))


#     def forward(self, inputs):
#         steps_out, _ = self.model.encoder.network(inputs)
#         main_out = self.model.decoder(steps_out)
#         recon_out = self.recon_head(steps_out)
#         return recon_out, main_out



# class TabTransformer(nn.Module):
#     def __init__(self, args, dataset):
#         print(f"dir(TabTransformer): {dir(TabTransformer.__init__)}")
#         self.model = TabTransformer(
#             categories=(10, 5, 6, 5, 8),      # tuple containing the number of unique values within each category
#             num_continuous=dataset.in_dim,                # number of continuous values
#             dim=32,                           # dimension, paper set at 32
#             dim_out=dataset.out_dim,                        # binary prediction, but could be anything
#             depth=6,                          # depth, paper recommended 6
#             heads=8,                          # heads, paper recommends 8
#             attn_dropout=0.1,                 # post-attention dropout
#             ff_dropout=0.1,                   # feed forward dropout
#             mlp_hidden_mults=(4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
#             mlp_act=nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
#             # continuous_mean_std=cont_mean_std # (optional) - normalize the continuous values before layer norm
#         )
#         print(f"dir(self.model): {dir(self.model)}")
#         self.recon_head = nn.Sequential()
#         self.recon_head.append(nn.Linear(256, dataset.in_dim))


#     def forward(self, inputs):
#         hidden_repr = self.model(inputs)
#         recon_out = self.recon_head(hidden_repr)
#         main_out = self.main_head(hidden_repr)