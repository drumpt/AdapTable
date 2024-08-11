import omegaconf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from pytorch_tabnet.tab_network import TabNetEncoder, TabNetDecoder, EmbeddingGenerator, RandomObfuscator
from tab_transformer_pytorch.tab_transformer_pytorch import Transformer as TabTransformerBlock
from tab_transformer_pytorch.ft_transformer import NumericalEmbedder, Transformer as FTTransformerBlock
from tab_transformer_pytorch.tab_transformer_pytorch import MLP as TabTransformerMLP
# from rtdl_revisiting_models import ResNet as RTDLResNet
# import sys
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/tableshift"))
# from tableshift.models.utils import get_estimator



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
        self.encoder = []
        for layer_idx, hidden_dim in zip(range(args.mlp.num_enc_layers - 1), hidden_dim_list):
            self.encoder.extend([
                nn.Linear(input_dim if layer_idx == 0 else hidden_dim_list[layer_idx - 1], hidden_dim),
                nn.BatchNorm1d(hidden_dim) if args.mlp.bn else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(args.mlp.dropout_rate),
            ])
        self.encoder.append(nn.Linear(hidden_dim_list[-2], hidden_dim_list[-1]))
        self.encoder = nn.Sequential(*self.encoder)

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


    # def forward(self, inputs):
    #     def get_embedding(inputs):
    #         if self.embedding and len(self.emb_layers):
    #             inputs_cont = inputs[:, :self.cat_start_index]
    #             inputs_cat = inputs[:, self.cat_start_index:]
    #             inputs_cat_emb = []
    #             for i, emb_layer in enumerate(self.emb_layers):
    #                 inputs_cat_emb.append(emb_layer(torch.argmax(inputs_cat[:, self.cat_start_indices[i]:self.cat_end_indices[i]], dim=-1)))
    #             inputs_cat = torch.cat(inputs_cat_emb, dim=-1)
    #             inputs = torch.cat([inputs_cont, inputs_cat], 1)
    #         return inputs

    #     inputs = get_embedding(inputs)
    #     print(f"inputs.device: {inputs.device}")
        
    #     hidden_repr = self.encoder(inputs)
    #     outputs = self.main_head(hidden_repr)
    #     return outputs


    def get_recon_out(self, inputs):
        inputs = self.get_embedding(inputs)
        hidden_repr = self.encoder(inputs)
        recon_out = self.recon_head(hidden_repr)
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

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.main_head = nn.Sequential(
            nn.Linear(hidden_dim + output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, shifted_inputs, vanilla_outputs):
        t = self.get_temperature(shifted_inputs, vanilla_outputs)
        return torch.mul(vanilla_outputs, t)


    def get_temperature(self, shifted_inputs, vanilla_outputs):
        out = self.input_layer(shifted_inputs)
        t = self.main_head(torch.cat([out, vanilla_outputs], dim=-1))
        t = F.softplus(t)
        return t





from warnings import warn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import nn_utils
from .nn_utils import sparsemax, sparsemoid, ModuleWithInit


def check_numpy(x):
    """ Makes sure x is a numpy array """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x



class ODST(ModuleWithInit):
    def __init__(self, in_features, num_trees, depth=6, tree_dim=1, flatten_output=True,
                 choice_function=sparsemax, bin_function=sparsemoid,
                 initialize_response_=nn.init.normal_, initialize_selection_logits_=nn.init.uniform_,
                 threshold_init_beta=1.0, threshold_init_cutoff=1.0,
                 ):
        """
        Oblivious Differentiable Sparsemax Trees. http://tinyurl.com/odst-readmore
        One can drop (sic!) this module anywhere instead of nn.Linear
        :param in_features: number of features in the input tensor
        :param num_trees: number of trees in this layer
        :param tree_dim: number of response channels in the response of individual tree
        :param depth: number of splits in every tree
        :param flatten_output: if False, returns [..., num_trees, tree_dim],
            by default returns [..., num_trees * tree_dim]
        :param choice_function: f(tensor, dim) -> R_simplex computes feature weights s.t. f(tensor, dim).sum(dim) == 1
        :param bin_function: f(tensor) -> R[0, 1], computes tree leaf weights

        :param initialize_response_: in-place initializer for tree output tensor
        :param initialize_selection_logits_: in-place initializer for logits that select features for the tree
        both thresholds and scales are initialized with data-aware init (or .load_state_dict)
        :param threshold_init_beta: initializes threshold to a q-th quantile of data points
            where q ~ Beta(:threshold_init_beta:, :threshold_init_beta:)
            If this param is set to 1, initial thresholds will have the same distribution as data points
            If greater than 1 (e.g. 10), thresholds will be closer to median data value
            If less than 1 (e.g. 0.1), thresholds will approach min/max data values.

        :param threshold_init_cutoff: threshold log-temperatures initializer, \in (0, inf)
            By default(1.0), log-remperatures are initialized in such a way that all bin selectors
            end up in the linear region of sparse-sigmoid. The temperatures are then scaled by this parameter.
            Setting this value > 1.0 will result in some margin between data points and sparse-sigmoid cutoff value
            Setting this value < 1.0 will cause (1 - value) part of data points to end up in flat sparse-sigmoid region
            For instance, threshold_init_cutoff = 0.9 will set 10% points equal to 0.0 or 1.0
            Setting this value > 1.0 will result in a margin between data points and sparse-sigmoid cutoff value
            All points will be between (0.5 - 0.5 / threshold_init_cutoff) and (0.5 + 0.5 / threshold_init_cutoff)
        """
        super().__init__()
        self.depth, self.num_trees, self.tree_dim, self.flatten_output = depth, num_trees, tree_dim, flatten_output
        self.choice_function, self.bin_function = choice_function, bin_function
        self.threshold_init_beta, self.threshold_init_cutoff = threshold_init_beta, threshold_init_cutoff

        self.response = nn.Parameter(torch.zeros([num_trees, tree_dim, 2 ** depth]), requires_grad=True)
        initialize_response_(self.response)

        self.feature_selection_logits = nn.Parameter(
            torch.zeros([in_features, num_trees, depth]), requires_grad=True
        )
        initialize_selection_logits_(self.feature_selection_logits)

        self.feature_thresholds = nn.Parameter(
            torch.full([num_trees, depth], float('nan'), dtype=torch.float32), requires_grad=True
        )  # nan values will be initialized on first batch (data-aware init)

        self.log_temperatures = nn.Parameter(
            torch.full([num_trees, depth], float('nan'), dtype=torch.float32), requires_grad=True
        )

        # binary codes for mapping between 1-hot vectors and bin indices
        with torch.no_grad():
            indices = torch.arange(2 ** self.depth)
            offsets = 2 ** torch.arange(self.depth)
            bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(torch.float32)
            bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
            self.bin_codes_1hot = nn.Parameter(bin_codes_1hot, requires_grad=False)
            # ^-- [depth, 2 ** depth, 2]

        # print(f"{self.response=}")
        # print(f"{self.feature_selection_logits=}")
        print(f"{self.feature_thresholds.dtype=}")
        print(f"{self.log_temperatures.dtype=}")
        # print(f"{self.bin_codes_1hot=}")

    def forward(self, input):
        assert len(input.shape) >= 2
        if len(input.shape) > 2:
            return self.forward(input.view(-1, input.shape[-1])).view(*input.shape[:-1], -1)
        # new input shape: [batch_size, in_features]

        feature_logits = self.feature_selection_logits
        feature_selectors = self.choice_function(feature_logits, dim=0)
        # ^--[in_features, num_trees, depth]

        feature_values = torch.einsum('bi,ind->bnd', input, feature_selectors)
        # ^--[batch_size, num_trees, depth]

        threshold_logits = (feature_values - self.feature_thresholds) * torch.exp(-self.log_temperatures)

        threshold_logits = torch.stack([-threshold_logits, threshold_logits], dim=-1)
        # ^--[batch_size, num_trees, depth, 2]

        bins = self.bin_function(threshold_logits)
        # ^--[batch_size, num_trees, depth, 2], approximately binary

        bin_matches = torch.einsum('btds,dcs->btdc', bins, self.bin_codes_1hot)
        # ^--[batch_size, num_trees, depth, 2 ** depth]

        response_weights = torch.prod(bin_matches, dim=-2)
        # ^-- [batch_size, num_trees, 2 ** depth]

        response = torch.einsum('bnd,ncd->bnc', response_weights, self.response)
        # ^-- [batch_size, num_trees, tree_dim]

        return response.flatten(1, 2) if self.flatten_output else response

    def initialize(self, input, eps=1e-6):
        # data-aware initializer
        assert len(input.shape) == 2
        if input.shape[0] < 1000:
            warn("Data-aware initialization is performed on less than 1000 data points. This may cause instability."
                 "To avoid potential problems, run this model on a data batch with at least 1000 data samples."
                 "You can do so manually before training. Use with torch.no_grad() for memory efficiency.")
        with torch.no_grad():
            feature_selectors = self.choice_function(self.feature_selection_logits, dim=0)
            # ^--[in_features, num_trees, depth]

            feature_values = torch.einsum('bi,ind->bnd', input, feature_selectors)
            # ^--[batch_size, num_trees, depth]

            # initialize thresholds: sample random percentiles of data
            percentiles_q = 100 * np.random.beta(self.threshold_init_beta, self.threshold_init_beta,
                                                 size=[self.num_trees, self.depth])
            self.feature_thresholds.data[...] = torch.as_tensor(
                list(map(np.percentile, check_numpy(feature_values.flatten(1, 2).t()), percentiles_q.flatten())),
                dtype=feature_values.dtype, device=feature_values.device
            ).view(self.num_trees, self.depth)

            # init temperatures: make sure enough data points are in the linear region of sparse-sigmoid
            temperatures = np.percentile(check_numpy(abs(feature_values - self.feature_thresholds)),
                                         q=100 * min(1.0, self.threshold_init_cutoff), axis=0)

            # if threshold_init_cutoff > 1, scale everything down by it
            temperatures /= max(1.0, self.threshold_init_cutoff)
            self.log_temperatures.data[...] = torch.log(torch.as_tensor(temperatures) + eps)

    def __repr__(self):
        return "{}(in_features={}, num_trees={}, depth={}, tree_dim={}, flatten_output={})".format(
            self.__class__.__name__, self.feature_selection_logits.shape[0],
            self.num_trees, self.depth, self.tree_dim, self.flatten_output
        )



class DenseBlock(nn.Sequential):
    def __init__(self, input_dim, layer_dim, num_layers, tree_dim=1, max_features=None,
                 input_dropout=0.0, flatten_output=True, Module=ODST, **kwargs):
        layers = []
        for i in range(num_layers):
            oddt = Module(input_dim, layer_dim, tree_dim=tree_dim, flatten_output=True, **kwargs)
            input_dim = min(input_dim + layer_dim * tree_dim, max_features or float('inf'))
            layers.append(oddt)

        super().__init__(*layers)
        self.num_layers, self.layer_dim, self.tree_dim = num_layers, layer_dim, tree_dim
        self.max_features, self.flatten_output = max_features, flatten_output
        self.input_dropout = input_dropout

    def forward(self, x):
        initial_features = x.shape[-1]
        for layer in self:
            layer_inp = x
            if self.max_features is not None:
                tail_features = min(self.max_features, layer_inp.shape[-1]) - initial_features
                if tail_features != 0:
                    layer_inp = torch.cat([layer_inp[..., :initial_features], layer_inp[..., -tail_features:]], dim=-1)
            if self.training and self.input_dropout:
                layer_inp = F.dropout(layer_inp, self.input_dropout)
            h = layer(layer_inp)
            x = torch.cat([x, h], dim=-1)

        outputs = x[..., initial_features:]
        if not self.flatten_output:
            outputs = outputs.view(*outputs.shape[:-1], self.num_layers * self.layer_dim, self.tree_dim)
        return outputs



class NODE(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()

        self.cat_start_index = dataset.cont_dim
        self.cat_end_indices = np.cumsum([num_category for num_category, _ in dataset.emb_dim_list])
        self.cat_start_indices = np.concatenate([[0], self.cat_end_indices], axis=0)[:-1]
        self.cat_indices_groups = dataset.cat_indices_groups

        self.in_features = dataset.cont_dim + len(dataset.emb_dim_list)
        self.dim_out = dataset.out_dim

        self.encoder = DenseBlock(
            self.in_features,
            2048,
            num_layers=1,
            tree_dim=3,
            depth=6,
            flatten_output=False,
            choice_function=nn_utils.entmax15,
            bin_function=nn_utils.entmoid15
        )
        print(f"{self.encoder=}")
        self.lda = nn_utils.Lambda(lambda x: x[..., :self.dim_out].mean(dim=-2))


    def forward(self, inputs):
        inputs = self.get_le_from_oe(inputs)
        x = self.encoder(inputs)
        outputs = self.lda(x)
        return outputs


    # def get_recon_out(self, inputs):
    #     inputs = self.get_embedding(inputs)
    #     x = self.transformer(inputs, return_attn=False)
    #     feature_out = x[:, 0]
    #     recon_out = self.recon_head(feature_out)
    #     return recon_out

    
    # def get_feature(self, inputs):
    #     inputs = self.get_embedding(inputs)
    #     x = self.transformer(inputs, return_attn=False)
    #     feature_out = x[:, 0]
    #     return feature_out


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



import math
import typing as ty

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def reglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


def get_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return (
        reglu
        if name == 'reglu'
        else geglu
        if name == 'geglu'
        else torch.sigmoid
        if name == 'sigmoid'
        else getattr(F, name)
    )


def get_nonglu_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return (
        F.relu
        if name == 'reglu'
        else F.gelu
        if name == 'geglu'
        else get_activation_fn(name)
    )

# %%
class RTDLResNet(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
        d: int,
        d_hidden_factor: float,
        n_layers: int,
        activation: str,
        normalization: str,
        hidden_dropout: float,
        residual_dropout: float,
        d_out: int,
    ) -> None:
        super().__init__()

        def make_normalization():
            return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
                normalization
            ](d)

        self.main_activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout

        d_in = d_numerical
        d_hidden = int(d * d_hidden_factor)

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'norm': make_normalization(),
                        'linear0': nn.Linear(
                            d, d_hidden * (2 if activation.endswith('glu') else 1)
                        ),
                        'linear1': nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, d_out)

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)

        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x
    
    def get_feature(self, inputs):
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)

        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        return x



class ResNet(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()

        self.cat_start_index = dataset.cont_dim
        self.cat_end_indices = np.cumsum([num_category for num_category, _ in dataset.emb_dim_list])
        self.cat_start_indices = np.concatenate([[0], self.cat_end_indices], axis=0)[:-1]
        self.cat_indices_groups = dataset.cat_indices_groups

        input_dim = dataset.in_dim
        output_dim = dataset.out_dim

        self.model = RTDLResNet(
            d_in=input_dim,
            d_out=output_dim,
            n_blocks=2,
            d_block=192,
            d_hidden=None,
            d_hidden_multiplier=2.0,
            dropout1=0.15,
            dropout2=0.0,
        )

    def forward(self, inputs, graph_embedding=None):
        outputs = self.model(inputs)
        return outputs

    def get_feature(self, inputs):
        features = self.model.get_feature(inputs)
        return outputs


import math
import typing as ty

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor



class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        n_latent_tokens: int,
        d_token: int,
    ) -> None:
        super().__init__()
        assert n_latent_tokens == 0
        self.n_latent_tokens = n_latent_tokens
        if d_numerical:
            self.weight = nn.Parameter(Tensor(d_numerical + n_latent_tokens, d_token))
            # The initialization is inspired by nn.Linear
            nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            self.weight = None
            assert categories is not None
        if categories is None:
            self.category_offsets = None
            self.category_embeddings = None
        else:
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

    @property
    def n_tokens(self) -> int:
        return (0 if self.weight is None else len(self.weight)) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        if x_num is None:
            return self.category_embeddings(x_cat + self.category_offsets[None])  # type: ignore[code]
        x_num = torch.cat(
            [
                torch.ones(len(x_num), self.n_latent_tokens, device=x_num.device),
                x_num,
            ],
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]  # type: ignore[code]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],  # type: ignore[code]
                dim=1,
            )
        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, initialization: str
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear],
    ) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class RTDLAutoInt(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        n_layers: int,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        d_out: int,
    ) -> None:
        assert not prenormalization
        assert activation == 'relu'
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)

        super().__init__()
        self.tokenizer = Tokenizer(d_numerical, categories, 0, d_token)
        n_tokens = self.tokenizer.n_tokens

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear': nn.Linear(d_token, d_token, bias=False),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.activation = get_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token * n_tokens, d_out)

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        x = self.tokenizer(x_num, x_cat)

        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                x_residual,
                x_residual,
                *self._get_kv_compressions(layer),
            )
            x = layer['linear'](x)
            x = self._end_residual(x, x_residual, layer, 0)
            x = self.activation(x)

        x = x.flatten(1, 2)
        x = self.head(x)
        x = x.squeeze(-1)
        return x


class AutoInt(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()

        self.cat_start_index = dataset.cont_dim
        self.cat_end_indices = np.cumsum([num_category for num_category, _ in dataset.emb_dim_list])
        self.cat_start_indices = np.concatenate([[0], self.cat_end_indices], axis=0)[:-1]
        self.cat_indices_groups = dataset.cat_indices_groups

        self.model = RTDLAutoInt(
            d_numerical=dataset.cont_dim,
            categories=[num_category for num_category, _ in dataset.emb_dim_list],
            d_out=dataset.out_dim,
            n_layers=3,
            d_token=62,
            n_heads=2,
            attention_dropout=0.1826903968910185,
            residual_dropout=0.0,
            activation='relu',
            prenormalization=False,
            initialization='kaiming',
            kv_compression=None,
            kv_compression_sharing=None,
        )

    def forward(self, inputs, graph_embedding=None):
        if len(self.cat_start_indices):
            inputs_cont = inputs[:, :self.cat_start_index]
            inputs_cat = inputs[:, self.cat_start_index:]
            inputs_cat_emb = [] # translate one-hot encoding to label encoding
            for i in range(len(self.cat_end_indices)):
                inputs_cat_emb.append(torch.argmax(inputs_cat[:, self.cat_start_indices[i]:self.cat_end_indices[i]], dim=-1))
            inputs_cat = torch.stack(inputs_cat_emb, dim=-1)
        else:
            inputs_cont = inputs
            inputs_cat = None
        outputs = self.model(inputs_cont, inputs_cat)
        return outputs

    def get_feature(self, inputs):
        features = self.model.get_feature(inputs)
        return outputs