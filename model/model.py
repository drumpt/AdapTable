import omegaconf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from pytorch_tabnet.tab_network import TabNetEncoder, TabNetDecoder, EmbeddingGenerator, RandomObfuscator
from tab_transformer_pytorch.tab_transformer_pytorch import Transformer as TabTransformerBlock
from tab_transformer_pytorch.ft_transformer import NumericalEmbedder, Transformer as FTTransformerBlock
from tab_transformer_pytorch.tab_transformer_pytorch import MLP as TabTransformerMLP

from data.dataset import Dataset



class MLP(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()

        self.embedding = args.mlp.embedding
        self.cat_start_index = dataset.cont_dim
        self.cat_end_indices = np.cumsum([num_category for num_category, _ in dataset.emb_dim_list])
        self.cat_start_indices = np.concatenate([[0], self.cat_end_indices], axis=0)[:-1]
        self.cat_indices_groups = dataset.cat_indices_groups

        input_dim = dataset.in_dim if not self.embedding else dataset.cont_dim + sum([dim for _, dim in dataset.emb_dim_list])
        if isinstance(args.mlp.hidden_dim, list):
            assert len(args.mlp.hidden_dim) == num_enc_layers
        hidden_dim_list = args.mlp.hidden_dim if isinstance(args.mlp.hidden_dim, omegaconf.listconfig.ListConfig) else [args.mlp.hidden_dim for _ in range(args.mlp.num_enc_layers)]
        output_dim = dataset.out_dim

        if self.embedding:
            self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in dataset.emb_dim_list])
        self.encoder = nn.Sequential()
        for layer_idx, hidden_dim in zip(range(args.mlp.num_enc_layers - 1), hidden_dim_list):
            self.encoder.extend([
                nn.Linear(input_dim if layer_idx == 0 else hidden_dim_list[layer_idx - 1], hidden_dim),
                nn.BatchNorm1d(hidden_dim) if args.mlp.bn else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(args.mlp.dropout_rate),
            ])
        self.encoder.append(nn.Linear(hidden_dim_list[-2], hidden_dim_list[-1]))
        self.main_head = nn.Sequential(
            nn.Linear(hidden_dim_list[-1], hidden_dim_list[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dim_list[-1], output_dim),
        )
        self.recon_head = nn.Sequential(
            nn.Linear(hidden_dim_list[-1], hidden_dim_list[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dim_list[-1], dataset.in_dim),
        )


    def forward(self, inputs, graph_embedding=None):
        inputs = self.get_embedding(inputs)
        hidden_repr = self.encoder(inputs)
        if graph_embedding is not None:
            outputs = self.graph_embed_head(torch.cat([hidden_repr, graph_embedding.repeat(hidden_repr.shape[0], 1)], dim=-1))
        else:
            outputs = self.main_head(hidden_repr)
        return outputs


    def get_recon_out(self, inputs):
        inputs = self.get_embedding(inputs)
        hidden_repr = self.encoder(inputs)
        recon_out = self.recon_head(hidden_repr)
        # if len(self.cat_indices_groups):
        #     recon_out = Dataset.revert_recon_to_onehot(recon_out, self.cat_indices_groups)
        return recon_out


    def get_feature(self, inputs):
        inputs = self.get_embedding(inputs)
        hidden_repr = self.encoder(inputs)
        return hidden_repr


    def get_embedding(self, inputs):
        if self.embedding and len(self.emb_layers):
            inputs_cont = inputs[:, :self.cat_start_index]
            inputs_cat = inputs[:, self.cat_start_index:]
            inputs_cat_emb = []
            for i, emb_layer in enumerate(self.emb_layers):
                inputs_cat_emb.append(emb_layer(torch.argmax(inputs_cat[:, self.cat_start_indices[i]:self.cat_end_indices[i]], dim=-1)))
            inputs_cat = torch.cat(inputs_cat_emb, dim=-1)
            inputs = torch.cat([inputs_cont, inputs_cat], 1)
        return inputs



class TabNet(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()

        self.embedding = True
        self.input_dim = dataset.cont_dim + len(dataset.cat_start_indices)
        self.cat_start_index = dataset.cont_dim
        self.cat_end_indices = np.cumsum([emb_dim for _, emb_dim in dataset.emb_dim_list])
        self.cat_start_indices = np.concatenate([[0], self.cat_end_indices], axis=0)[:-1]
        self.cat_indices_groups = dataset.cat_indices_groups
        self.output_dim = dataset.out_dim

        self.pretraining_ratio = 0.2
        self.n_d = 8
        self.n_a = 8
        self.n_steps = 3
        self.gamma = 1.3
        self.epsilon = 1e-15
        self.n_independent = 2
        self.n_shared = 2
        self.mask_type = 'sparsemax'
        self.n_shared_decoder = 1
        self.n_indep_decoder = 1
        self.virtual_batch_size = 128
        self.momentum = 0.02
        self.embedding = True

        self.embedder = EmbeddingGenerator(
            input_dim=self.input_dim,
            cat_dims=list(dataset.cat_end_indices - dataset.cat_start_indices),
            cat_idxs=list(np.arange(dataset.cont_dim, dataset.cont_dim + len(dataset.emb_dim_list))),
            cat_emb_dims=[emb_dim for _, emb_dim in dataset.emb_dim_list],
            group_matrix=torch.eye(dataset.in_dim).to(args.device),
        )
        self.post_embed_dim = self.embedder.post_embed_dim
        self.encoder = TabNetEncoder(
            input_dim=self.post_embed_dim,
            output_dim=self.post_embed_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            epsilon=self.epsilon,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            mask_type=self.mask_type,
            group_attention_matrix=self.embedder.embedding_group_matrix,
        )
        self.decoder = TabNetDecoder(
            dataset.in_dim,
            n_d=self.n_d,
            n_steps=self.n_steps,
            n_independent=self.n_indep_decoder,
            n_shared=self.n_shared_decoder,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
        )
        self.main_head = nn.Sequential(
            nn.Linear(self.n_d, self.n_d),
            nn.ReLU(),
            nn.Linear(self.n_d, self.output_dim),
        )


    def forward(self, inputs):
        embedded_inputs = self.get_embedding(inputs)
        steps_out, _ = self.encoder(embedded_inputs)
        res = torch.sum(torch.stack(steps_out, dim=0), dim=0)
        outputs = self.main_head(res)
        return outputs


    def get_recon_out(self, inputs):
        embedded_inputs = self.get_embedding(inputs)
        steps_out, _ = self.encoder(embedded_inputs)
        recon_out = self.decoder(steps_out)
        # if len(self.cat_indices_groups):
        #     recon_out = Dataset.revert_recon_to_onehot(recon_out, self.cat_indices_groups)
        return recon_out


    def get_feature(self, inputs):
        embedded_inputs = self.get_embedding(inputs)
        steps_out, _ = self.encoder(embedded_inputs)
        steps_out = torch.sum(torch.stack(steps_out, dim=0), dim=0)
        return steps_out


    def get_embedding(self, inputs):
        inputs = self.get_le_from_oe(inputs)
        embedded_inputs = self.embedder(inputs)
        return embedded_inputs


    def get_le_from_oe(self, inputs):
        if len(self.cat_start_indices):
            inputs_cont = inputs[:, :self.cat_start_index]
            inputs_cat = inputs[:, self.cat_start_index:]
            inputs_cat_emb = [] # translate one-hot encoding to label encoding
            for i in range(len(self.cat_end_indices)):
                inputs_cat_emb.append(torch.argmax(inputs_cat[:, self.cat_start_indices[i]:self.cat_end_indices[i]], dim=-1))
            inputs_cat = torch.stack(inputs_cat_emb, dim=-1)
            inputs = torch.cat([inputs_cont, inputs_cat], dim=-1)
        return inputs



class TabTransformer(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()
        self.embedding = True
        self.cat_start_index = dataset.cont_dim
        self.cat_end_indices = np.cumsum([num_category for num_category, _ in dataset.emb_dim_list])
        self.cat_start_indices = np.concatenate([[0], self.cat_end_indices], axis=0)[:-1]
        self.cat_indices_groups = dataset.cat_indices_groups

        self.dim = 32 # dimension, paper set at 32
        self.depth = 6 # depth, paper recommended 6
        self.heads = 8 # heads, paper recommends 8
        self.dim_head = 16
        self.dim_out = dataset.out_dim
        self.attn_dropout = 0 # post-attention dropout
        self.ff_dropout = 0 # feed forward dropout
        self.mlp_hidden_mults = (4, 2) # relative multiples of each hidden dimension of the last mlp to logits

        categories = dataset.cat_end_indices - dataset.cat_start_indices
        num_continuous = dataset.cont_dim
        num_special_tokens = 2

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

        self.num_continuous = num_continuous
        if self.num_continuous > 0:
            self.norm = nn.LayerNorm(num_continuous)

        self.transformer = TabTransformerBlock(
            num_tokens=self.total_tokens,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            dim_head=self.dim_head,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout
        )

        input_size = (self.dim * self.num_categories) + num_continuous
        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, self.mlp_hidden_mults))
        main_head_dimensions = [input_size, *hidden_dimensions, self.dim_out]
        recon_head_dimensions = [input_size] * (len(main_head_dimensions) - 1) + [dataset.in_dim]

        self.main_head = TabTransformerMLP(main_head_dimensions)
        self.recon_head = TabTransformerMLP(recon_head_dimensions)


    def forward(self, inputs):
        inputs_emb = self.get_embedding(inputs)
        outputs = self.main_head(inputs_emb)
        return outputs


    def get_recon_out(self, inputs):
        inputs_emb = self.get_embedding(inputs)
        recon_out = self.recon_head(inputs_emb)
        # if len(self.cat_indices_groups):
        #     recon_out = Dataset.revert_recon_to_onehot(recon_out, self.cat_indices_groups)
        return recon_out


    def get_feature(self, inputs):
        inputs_emb = self.get_embedding(inputs)
        return inputs_emb


    def get_embedding(self, inputs):
        inputs = self.get_le_from_oe(inputs)
        inputs_cont = inputs[:, :self.cat_start_index]
        inputs_cat = inputs[:, self.cat_start_index:]
        xs = []
        if self.num_unique_categories > 0:
            inputs_cat += self.categories_offset
            x = self.transformer(inputs_cat.long(), return_attn=False)
            flat_categ = x.flatten(1)
            xs.append(flat_categ)
        if self.num_continuous > 0:
            normed_cont = self.norm(inputs_cont)
            xs.append(normed_cont)
        embedded_inputs = torch.cat(xs, dim=-1)
        return embedded_inputs
    
    
    def get_le_from_oe(self, inputs): # one-hot encoding -> label encoding
        if len(self.cat_start_indices):
            inputs_cont = inputs[:, :self.cat_start_index]
            inputs_cat = inputs[:, self.cat_start_index:]
            inputs_cat_emb = [] # translate one-hot encoding to label encoding
            for i in range(len(self.cat_end_indices)):
                inputs_cat_emb.append(torch.argmax(inputs_cat[:, self.cat_start_indices[i]:self.cat_end_indices[i]], dim=-1))
            inputs_cat = torch.stack(inputs_cat_emb, dim=-1)
            inputs = torch.cat([inputs_cont, inputs_cat], dim=-1)
        return inputs



class FTTransformer(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()

        self.embedding = True
        self.cat_start_index = dataset.cont_dim
        self.cat_end_indices = np.cumsum([num_category for num_category, _ in dataset.emb_dim_list])
        self.cat_start_indices = np.concatenate([[0], self.cat_end_indices], axis=0)[:-1]
        self.cat_indices_groups = dataset.cat_indices_groups

        self.dim = 32
        self.depth = 6
        self.heads = 8
        self.dim_head = 16
        self.dim_out = dataset.out_dim
        self.attn_dropout = 0
        self.ff_dropout = 0

        categories = dataset.cat_end_indices - dataset.cat_start_indices
        num_continuous = dataset.cont_dim
        num_special_tokens = 2

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)
            self.categorical_embeds = nn.Embedding(total_tokens, self.dim)

        self.num_continuous = num_continuous
        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(self.dim, self.num_continuous)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))

        self.transformer = FTTransformerBlock(
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            dim_head=self.dim_head,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
        )
        self.main_head = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim_out)
        )
        self.recon_head = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, dataset.in_dim)
        )


    def forward(self, inputs):
        inputs = self.get_embedding(inputs)
        x = self.transformer(inputs, return_attn=False)
        feature_out = x[:, 0]
        outputs = self.main_head(feature_out)
        return outputs


    def get_recon_out(self, inputs):
        inputs = self.get_embedding(inputs)
        x = self.transformer(inputs, return_attn=False)
        feature_out = x[:, 0]
        recon_out = self.recon_head(feature_out)
        # if len(self.cat_indices_groups):
        #     recon_out = Dataset.revert_recon_to_onehot(recon_out, self.cat_indices_groups)
        return recon_out

    
    def get_feature(self, inputs):
        inputs = self.get_embedding(inputs)
        x = self.transformer(inputs, return_attn=False)
        feature_out = x[:, 0]
        return feature_out


    def get_embedding(self, inputs):
        inputs = self.get_le_from_oe(inputs)
        inputs_cont = inputs[:, :self.cat_start_index]
        inputs_cat = inputs[:, self.cat_start_index:]
        xs = []
        if self.num_unique_categories > 0:
            inputs_cat = inputs_cat + self.categories_offset
            inputs_cat = self.categorical_embeds(inputs_cat.long())
            xs.append(inputs_cat)
        if self.num_continuous > 0:
            inputs_cont = self.numerical_embedder(inputs_cont)
            xs.append(inputs_cont)
        x = torch.cat(xs, dim=1)
        b = x.shape[0] # append cls tokens
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        return x


    def get_le_from_oe(self, inputs): # one-hot encoding -> label encoding
        if len(self.cat_start_indices):
            inputs_cont = inputs[:, :self.cat_start_index]
            inputs_cat = inputs[:, self.cat_start_index:]
            inputs_cat_emb = [] # translate one-hot encoding to label encoding
            for i in range(len(self.cat_end_indices)):
                inputs_cat_emb.append(torch.argmax(inputs_cat[:, self.cat_start_indices[i]:self.cat_end_indices[i]], dim=-1))
            inputs_cat = torch.stack(inputs_cat_emb, dim=-1)
            inputs = torch.cat([inputs_cont, inputs_cat], dim=-1)
        return inputs



class ColumnShiftHandler(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()
        input_dim = dataset.in_dim
        hidden_dim = 16
        output_dim = dataset.out_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.main_head = nn.Linear(hidden_dim + output_dim, output_dim)
        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, output_dim, bias=False),
        # )


    def forward(self, inputs, vanilla_outputs):
        outputs = self.model(inputs)
        outputs = outputs.repeat(len(vanilla_outputs), 1)
        # print(f"outputs: {outputs}")
        # print(f"outputs.shape: {outputs.shape}")
        return self.main_head(torch.cat([outputs, vanilla_outputs], dim=-1))
        # outputs = self.model(inputs)
        # outputs = outputs.repeat(len(vanilla_outputs), 1)
        # # print(f"outputs: {outputs}")
        # # print(f"outputs.shape: {outputs.shape}")
        # return outputs + vanilla_outputs