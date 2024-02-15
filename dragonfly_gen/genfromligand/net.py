#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (c) 2023 Kenneth Atz, Clemens Isert & Gisbert Schneider (ETH Zurich)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, Size, Tensor

from dragonfly_gen.gml.pygmt import GraphMultisetTransformer, scatter_sum


class LSTM(nn.Module):
    def __init__(
        self, input_dim=57, hidden_dim=256, layers=2, dropout=0.3, token_embedding=128
    ):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim
        self.embedding_dimension = token_embedding
        self.layers = layers
        self.dropout = dropout

        self.token_embedding = nn.Embedding(self.input_dim, self.embedding_dimension)

        self.lstm = nn.LSTM(
            input_size=self.embedding_dimension,
            hidden_size=self.hidden_dim,
            num_layers=self.layers,
            dropout=self.dropout,
        )

        self.norm_0 = nn.LayerNorm(self.embedding_dimension, eps=0.001)
        self.norm_1 = nn.LayerNorm(self.hidden_dim, eps=0.001)
        self.fnn = nn.Linear(self.hidden_dim, self.output_dim)

        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l1)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l1)

        self.lstm.bias_ih_l0.data.fill_(0.0)
        self.lstm.bias_ih_l0.data[self.hidden_dim : 2 * self.hidden_dim].fill_(1.0)
        self.lstm.bias_ih_l1.data.fill_(0.0)
        self.lstm.bias_ih_l1.data[self.hidden_dim : 2 * self.hidden_dim].fill_(1.0)
        self.lstm.bias_hh_l0.data.fill_(0.0)
        self.lstm.bias_hh_l0.data[self.hidden_dim : 2 * self.hidden_dim].fill_(1.0)
        self.lstm.bias_hh_l1.data.fill_(0.0)
        self.lstm.bias_hh_l1.data[self.hidden_dim : 2 * self.hidden_dim].fill_(1.0)

        nn.init.xavier_uniform_(self.fnn.weight)
        nn.init.zeros_(self.fnn.bias)

        # self.act = nn.LogSoftmax(dim=2)
        # self.act = nn.Softmax(dim=2)

    def forward(self, input, hiddens):
        features = self.token_embedding(input)
        features = self.norm_0(features)
        features, hiddens = self.lstm(features, hiddens)
        features = self.norm_1(features)
        features = self.fnn(features)
        # features = self.act(features)

        return features, hiddens


class GraphTransformer(nn.Module):
    def __init__(self, n_kernels=3, rnn_dim=1024, property_dim=6, pooling_heads=8):
        super(GraphTransformer, self).__init__()

        self.num_embeddings_atom = 22
        self.num_embeddings_residue = 255
        self.embeddings_dim = 64
        self.pdb_prop_dim = 32
        self.m_dim = 16
        self.kernel_dim = 128
        self.n_kernels = n_kernels
        self.aggr = "add"
        self.pooling_heads = pooling_heads
        self.property = property_dim
        self.rnn_dim = rnn_dim

        dropout = 0.1
        self.dropout = nn.Dropout(dropout)

        self.atom_emb = nn.Embedding(
            num_embeddings=11, embedding_dim=self.embeddings_dim
        )
        self.is_ring_emb = nn.Embedding(
            num_embeddings=2, embedding_dim=self.embeddings_dim
        )
        self.hyb_emb = nn.Embedding(num_embeddings=4, embedding_dim=self.embeddings_dim)
        self.arom_emb = nn.Embedding(
            num_embeddings=2, embedding_dim=self.embeddings_dim
        )

        self.pre_egnn_mlp = nn.Sequential(
            nn.Linear(self.embeddings_dim * 4, self.kernel_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.kernel_dim * 2, self.kernel_dim),
        )

        self.kernels = nn.ModuleList()
        for _ in range(self.n_kernels):
            self.kernels.append(
                EGNN_sparse(
                    feats_dim=self.kernel_dim,
                    m_dim=self.m_dim,
                    aggr=self.aggr,
                )
            )

        self.post_egnn_mlp = nn.Sequential(
            nn.Linear(self.kernel_dim * self.n_kernels, self.kernel_dim),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.kernel_dim, self.kernel_dim),
            nn.SiLU(),
            nn.Linear(self.kernel_dim, self.kernel_dim),
            nn.SiLU(),
        )

        self.transformers = nn.ModuleList()
        for _ in range(self.pooling_heads):
            self.transformers.append(
                GraphMultisetTransformer(
                    in_channels=self.kernel_dim,
                    hidden_channels=self.kernel_dim,
                    out_channels=self.kernel_dim,
                    pool_sequences=["GMPool_G", "SelfAtt", "GMPool_I"],
                    num_heads=1,
                    layer_norm=True,
                )
            )

        if self.property == 6:
            self.mol_property_lin = nn.Linear(self.property, self.kernel_dim)
            self.mlp_input_dim = self.kernel_dim * (self.pooling_heads + 1)
            self.mol_property_lin.apply(weights_init)
        elif self.property == 1:
            self.mol_property_lin = nn.Linear(self.property, self.kernel_dim)
            self.mlp_input_dim = self.kernel_dim * (self.pooling_heads + 1)
            self.mol_property_lin.apply(weights_init)
        else:
            self.mlp_input_dim = self.kernel_dim * self.pooling_heads

        self.post_pooling_mlps = nn.ModuleList()
        for _ in range(2):
            self.post_pooling_mlps.append(
                nn.Sequential(
                    nn.Linear(self.mlp_input_dim, self.rnn_dim),
                    self.dropout,
                    nn.SiLU(),
                    nn.Linear(self.rnn_dim, self.rnn_dim),
                    nn.SiLU(),
                    nn.Linear(self.rnn_dim, self.rnn_dim),
                )
            )

        self.transformers.apply(weights_init)
        self.kernels.apply(weights_init)
        self.post_egnn_mlp.apply(weights_init)
        self.post_pooling_mlps.apply(weights_init)
        nn.init.xavier_uniform_(self.atom_emb.weight)
        nn.init.xavier_uniform_(self.is_ring_emb.weight)
        nn.init.xavier_uniform_(self.hyb_emb.weight)
        nn.init.xavier_uniform_(self.arom_emb.weight)

    def forward(self, g_batch):
        features = self.pre_egnn_mlp(
            torch.cat(
                [
                    self.atom_emb(g_batch.atomids),
                    self.is_ring_emb(g_batch.is_ring),
                    self.hyb_emb(g_batch.hyb),
                    self.arom_emb(g_batch.arom),
                ],
                dim=1,
            )
        )

        feature_list = []
        for kernel in self.kernels:
            feature_list.append(kernel(x=features, edge_index=g_batch.edge_index))

        features = torch.cat(feature_list, dim=1)
        features = self.post_egnn_mlp(features)

        feature_list = []
        for transformer in self.transformers:
            feature_list.append(
                transformer(
                    x=features, batch=g_batch.batch, edge_index=g_batch.edge_index
                )
            )

        features = torch.cat(feature_list, dim=1)

        if self.property == 6:
            features = torch.cat(
                [features, self.mol_property_lin(g_batch.properties)], dim=1
            )
        elif self.property == 1:
            features = torch.cat([features, self.mol_property_lin(g_batch.sim)], dim=1)

        feature_list = []
        for mlp in self.post_pooling_mlps:
            feature_list.append(mlp(features).unsqueeze(0))

        features = torch.cat(feature_list, dim=0)
        del feature_list

        features = (
            features,
            torch.zeros(2, features.size(1), features.size(2)).to(features.device),
        )

        return features


class EGNN(nn.Module):
    def __init__(self, n_kernels=3, rnn_dim=1024, property_dim=6):
        super(EGNN, self).__init__()

        self.embeddings_dim = 64
        self.pdb_prop_dim = 32
        self.m_dim = 16
        self.kernel_dim = 128
        self.n_kernels = n_kernels
        self.aggr = "add"
        self.property = property_dim
        self.rnn_dim = rnn_dim

        dropout = 0.1
        self.dropout = nn.Dropout(dropout)

        self.atom_emb = nn.Embedding(
            num_embeddings=11, embedding_dim=self.embeddings_dim
        )
        self.is_ring_emb = nn.Embedding(
            num_embeddings=2, embedding_dim=self.embeddings_dim
        )
        self.hyb_emb = nn.Embedding(num_embeddings=4, embedding_dim=self.embeddings_dim)
        self.arom_emb = nn.Embedding(
            num_embeddings=2, embedding_dim=self.embeddings_dim
        )

        self.pre_egnn_mlp = nn.Sequential(
            nn.Linear(self.embeddings_dim * 4, self.kernel_dim),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.kernel_dim, self.kernel_dim),
            nn.SiLU(),
            nn.Linear(self.kernel_dim, self.kernel_dim),
        )

        self.kernels = nn.ModuleList()
        for _ in range(self.n_kernels):
            self.kernels.append(
                EGNN_sparse(
                    feats_dim=self.kernel_dim,
                    m_dim=self.m_dim,
                    aggr=self.aggr,
                )
            )

        self.post_egnn_mlp = nn.Sequential(
            nn.Linear(self.kernel_dim * self.n_kernels, self.rnn_dim),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.rnn_dim, self.rnn_dim),
            nn.SiLU(),
            nn.Linear(self.rnn_dim, self.rnn_dim),
            nn.SiLU(),
        )

        if self.property == 6:
            self.mol_property_lin = nn.Linear(self.property, self.kernel_dim)
            self.mlp_input_dim = self.rnn_dim + self.kernel_dim
            self.mol_property_lin.apply(weights_init)
        elif self.property == 1:
            self.mol_property_lin = nn.Linear(self.property, self.kernel_dim)
            self.mlp_input_dim = self.rnn_dim + self.kernel_dim
            self.mol_property_lin.apply(weights_init)
        else:
            self.mlp_input_dim = self.rnn_dim

        self.post_pooling_mlps = nn.ModuleList()
        for _ in range(2):
            self.post_pooling_mlps.append(
                nn.Sequential(
                    nn.Linear(self.mlp_input_dim, self.rnn_dim),
                    self.dropout,
                    nn.SiLU(),
                    nn.Linear(self.rnn_dim, self.rnn_dim),
                    nn.SiLU(),
                    nn.Linear(self.rnn_dim, self.rnn_dim),
                )
            )

        self.kernels.apply(weights_init)
        self.pre_egnn_mlp.apply(weights_init)
        self.post_egnn_mlp.apply(weights_init)
        self.post_pooling_mlps.apply(weights_init)
        nn.init.xavier_uniform_(self.atom_emb.weight)
        nn.init.xavier_uniform_(self.is_ring_emb.weight)
        nn.init.xavier_uniform_(self.hyb_emb.weight)
        nn.init.xavier_uniform_(self.arom_emb.weight)

    def forward(self, g_batch):
        features = self.pre_egnn_mlp(
            torch.cat(
                [
                    self.atom_emb(g_batch.atomids),
                    self.is_ring_emb(g_batch.is_ring),
                    self.hyb_emb(g_batch.hyb),
                    self.arom_emb(g_batch.arom),
                ],
                dim=1,
            )
        )

        feature_list = []
        for kernel in self.kernels:
            feature_list.append(kernel(x=features, edge_index=g_batch.edge_index))

        features = torch.cat(feature_list, dim=1)
        features = self.post_egnn_mlp(features)
        features = scatter_sum(features, g_batch.batch, dim=0)

        if self.property == 6:
            features = torch.cat(
                [features, self.mol_property_lin(g_batch.properties)], dim=1
            )
        elif self.property == 1:
            features = torch.cat([features, self.mol_property_lin(g_batch.sim)], dim=1)

        feature_list = []
        for mlp in self.post_pooling_mlps:
            feature_list.append(mlp(features).unsqueeze(0))

        features = torch.cat(feature_list, dim=0)

        del feature_list

        features = (
            features,
            torch.zeros(2, features.size(1), features.size(2)).to(features.device),
        )

        return features


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class EGNN_sparse(MessagePassing):
    """torch geometric message-passing layer for 2D molecular graphs."""

    def __init__(self, feats_dim, m_dim=32, dropout=0.1, aggr="add", **kwargs):
        """Initialization of the 2D message passing layer.

        :param feats_dim: Node feature dimension.
        :type feats_dim: int
        :param m_dim: Meessage passing feature dimesnion, defaults to 32
        :type m_dim: int, optional
        :param dropout: Dropout value, defaults to 0.1
        :type dropout: float, optional
        :param aggr: Message aggregation type, defaults to "add"
        :type aggr: str, optional
        """
        assert aggr in {
            "add",
            "sum",
            "max",
            "mean",
        }, "pool method must be a valid option"

        kwargs.setdefault("aggr", aggr)
        super(EGNN_sparse, self).__init__(**kwargs)

        self.feats_dim = feats_dim
        self.m_dim = m_dim

        self.edge_input_dim = feats_dim * 2

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.edge_norm1 = nn.LayerNorm(m_dim)
        self.edge_norm2 = nn.LayerNorm(m_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            nn.SiLU(),
        )

        self.node_norm1 = nn.LayerNorm(feats_dim)
        self.node_norm2 = nn.LayerNorm(feats_dim)

        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        )

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
    ):
        """Forward pass in the mesaage passing fucntion.

        :param x: Node features.
        :type x: Tensor
        :param edge_index: Edge indices.
        :type edge_index: Adj
        :return: Updated node features.
        :rtype: Tensor
        """
        hidden_out = self.propagate(edge_index, x=x)

        return hidden_out

    def message(self, x_i, x_j):
        """Message passing.

        :param x_i: Node n_i.
        :type x_i: Tensor
        :param x_j: Node n_j.
        :type x_j: Tensor
        :return: Message m_ji
        :rtype: Tensor
        """
        m_ij = self.edge_mlp(torch.cat([x_i, x_j], dim=-1))
        return m_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """Overall propagation within the message passing.

        :param edge_index: Edge indices.
        :type edge_index: Adj
        :return: Updated node features.
        :rtype: Tensor
        """
        # get input tensors
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute("message", coll_dict)
        aggr_kwargs = self.inspector.distribute("aggregate", coll_dict)
        update_kwargs = self.inspector.distribute("update", coll_dict)

        # get messages
        m_ij = self.message(**msg_kwargs)
        m_ij = self.edge_norm1(m_ij)

        # aggregate messages
        m_i = self.aggregate(m_ij, **aggr_kwargs)
        m_i = self.edge_norm2(m_i)

        # get updated node features
        hidden_feats = self.node_norm1(kwargs["x"])
        hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
        hidden_out = self.node_norm2(hidden_out)
        hidden_out = kwargs["x"] + hidden_out

        return self.update((hidden_out), **update_kwargs)
