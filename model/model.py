import numpy as np
import torch
import torch.nn as nn

from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from tab_transformer_pytorch import TabTransformer



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, dropout=0.0):
        super().__init__()

        self.encoder = nn.Sequential()
        self.recon_head = nn.Sequential()
        self.main_head = nn.Sequential()
        self.cls_head = nn.Sequential()
        self.aux_head = nn.Sequential()

        in_dim = input_dim
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
        self.aux_head.append(nn.Linear(hidden_dim, hidden_dim))


    def forward(self, inputs):
        hidden_repr = self.encoder(inputs)
        main_out = self.cls_head(self.main_head(hidden_repr))
        return main_out


    def get_recon_out(self, inputs):
        hidden_repr = self.encoder(inputs)
        recon_out = self.recon_head(hidden_repr)
        return recon_out


    def get_feature(self, inputs):
        hidden_repr = self.encoder(inputs)
        feature = self.main_head(hidden_repr)
        return feature



class MLP_EMB(nn.Module):
    def __init__(self, input_dim, emb_dim, output_dim, hidden_dim, cat_start_index, n_layers, dropout=0.0):
        super().__init__()

        self.layers = nn.Sequential()
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y, _ in emb_dim])
        self.cls_layer = nn.Sequential()
        in_dim = cat_start_index + sum([dim for _, dim, _ in emb_dim])
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.cls_layer.append(nn.Linear(in_dim, output_dim))

        self.cat_start_index = cat_start_index
        self.cat_end_indices = np.cumsum([num_category for num_category, _, _ in emb_dim])
        self.cat_start_indices = np.concatenate([[0], self.cat_end_indices], axis=0)[:-1]


    def forward(self, inputs):
        inputs_cont = inputs[:, :self.cat_start_index]
        inputs_cat = inputs[:, self.cat_start_index:]
        inputs_cat_emb = []
        for i, emb_layer in enumerate(self.emb_layers):
            inputs_cat_emb.append(emb_layer(torch.argmax(inputs_cat[:, self.cat_start_indices[i]:self.cat_end_indices[i]], dim=-1)))
        inputs_cat = torch.cat(inputs_cat_emb, dim=-1)
        inputs = torch.cat([inputs_cont, inputs_cat], 1)
        outputs = self.cls_layer(self.layers(inputs))
        return outputs


    def get_feature(self, inputs):
        feature = self.layers(inputs)
        return feature



class TabNet(nn.Module): # TODO: (WIP) finish implementing this!
    def __init__(self, args, dataset):
        super().__init__()

        self.model = TabNetRegressor(
            device_name=args.device,
            optimizer_fn=getattr(torch.optim, 'AdamW'),
            optimizer_params=dict(lr=args.train_lr),
        ) if dataset.out_dim == 1 else TabNetClassifier(
            device_name=args.device,
            optimizer_fn=getattr(torch.optim, 'AdamW'),
            optimizer_params=dict(lr=args.train_lr),
        )
        self.recon_head = nn.Sequential()
        self.recon_head.append(nn.Linear(256, dataset.in_dim))


    def forward(self, inputs):
        steps_out, _ = self.model.encoder.network(inputs)
        recon_out = self.recon_head(steps_out)
        return recon_out


    def get_recon_out(self, inputs):
        steps_out, _ = self.model.encoder.network(inputs)
        main_out = self.model.decoder(steps_out)
        return main_out



class TabTransformer(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()

        self.model = TabTransformer(
            categories=(10, 5, 6, 5, 8),      # tuple containing the number of unique values within each category
            num_continuous=dataset.in_dim,                # number of continuous values
            dim=32,                           # dimension, paper set at 32
            dim_out=dataset.out_dim,                        # binary prediction, but could be anything
            depth=6,                          # depth, paper recommended 6
            heads=8,                          # heads, paper recommends 8
            attn_dropout=0.1,                 # post-attention dropout
            ff_dropout=0.1,                   # feed forward dropout
            mlp_hidden_mults=(4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
            mlp_act=nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
        )
        self.recon_head = nn.Sequential()
        self.recon_head.append(nn.Linear(256, dataset.in_dim))


    def forward(self, inputs):
        hidden_repr = self.model(inputs)
        recon_out = self.recon_head(hidden_repr)
        main_out = self.main_head(hidden_repr)