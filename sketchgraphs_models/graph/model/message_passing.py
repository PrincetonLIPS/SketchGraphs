"""This module contains the components of the graph model associated with handling
the message passing.
"""

import torch

from sketchgraphs_models import nn as sg_nn
import sketchgraphs_models.nn.functional
from sketchgraphs.pipeline import graph_model


class DenseSparsePreEmbedding(torch.nn.Module):
    """This is a generic pre-embedding module which combines dense and sparse pre-embeddings."""
    def __init__(self, target_type, feature_embeddings, fixed_embedding_cardinality, fixed_embedding_dim,
                 sparse_embedding_dim=None, embedding_dim=None):
        """Initializes a new DenseSparsePreEmbedding module.

        Parameters
        ----------
        target_type : enum
            The underlying enumeration indicating target types
        feature_embeddings : dict of modules
            A dictionary of embeddings for each of the sparse feature types.
        fixed_embedding_cardinality : int
            The number of classes in the fixed (dense) embedding layer.
        fixed_embedding_dim : int
            The dimension of the embedding for the fixed layer.
        sparse_embedding_dim : int, optional
            The dimension of the sparse embeddings. If None, assumed to be the same as the dense embedding.
        embedding_dim : int, optional
            The outpu dimension of the embedding. If None, assumed to be the same as the dense embedding.
        """
        super(DenseSparsePreEmbedding, self).__init__()
        sparse_embedding_dim = sparse_embedding_dim or fixed_embedding_dim
        embedding_dim = embedding_dim or fixed_embedding_dim

        self.target_type = target_type
        self.feature_embeddings = torch.nn.ModuleDict(feature_embeddings)
        self.sparse_embedding_dim = sparse_embedding_dim
        self.fixed_embedding_dim = fixed_embedding_dim
        self.fixed_embedding = torch.nn.Embedding(fixed_embedding_cardinality, fixed_embedding_dim)
        self.dense_merge = sg_nn.ConcatenateLinear(fixed_embedding_dim, sparse_embedding_dim, embedding_dim)

    def forward(self, fixed_features, sparse_features):
        fixed_embeddings = self.fixed_embedding(fixed_features)
        sparse_embeddings = fixed_embeddings.new_zeros((fixed_embeddings.shape[0], self.sparse_embedding_dim))

        for k, embedding_network in self.feature_embeddings.items():
            sf = sparse_features[self.target_type[k]]
            if sf is None or len(sf.index) == 0:
                continue

            assert (sf.index < fixed_embeddings.shape[0]).all()
            sparse_embeddings[sf.index] = embedding_network(sf.value)

        return self.dense_merge(fixed_embeddings, sparse_embeddings)


class DenseOnlyEmbedding(torch.nn.Module):
    """Generic pre-embedding module which encapsulates a pytorch embedding layer.

    This class is simply provided for compatibility with `DenseSparsePreEmbedding`, to construct
    models where sparse embeddings are not present.
    """
    def __init__(self, cardinality, dimension):
        super(DenseOnlyEmbedding, self).__init__()
        self.fixed_embedding = torch.nn.Embedding(cardinality, dimension)

    def forward(self, features, *_):
        return self.fixed_embedding(features)


class GraphModelCore(torch.nn.Module):
    """Component of the entity model used to compute global features, i.e. graph and node embeddings.

    This component is responsible for the computation that is independent of any
    specific target (edge / node). It is split off to ease sharing with sampling models.
    """
    def __init__(self, message_passing, node_embedding, edge_embedding, graph_embedding):
        super(GraphModelCore, self).__init__()
        self.message_passing = message_passing
        self.node_embedding = node_embedding
        self.edge_embedding = edge_embedding
        self.graph_embedding = graph_embedding

    def forward(self, data):
        graph = data['graph']

        with torch.autograd.profiler.record_function('feature_embedding'):
            node_pre_embedding = self.node_embedding(graph.node_features, graph.sparse_node_features)
            edge_pre_embedding = self.edge_embedding(graph.edge_features, graph.sparse_edge_features)

        node_post_embedding = self.message_passing(node_pre_embedding, graph.incidence, (edge_pre_embedding,))
        graph_embedding = self.graph_embedding(node_post_embedding, graph)
        return node_post_embedding, graph_embedding


class GraphPostEmbedding(torch.nn.Module):
    """Component of the graph model which computes graph-wide representation by aggregating node representations.
    """
    def __init__(self, hidden_size, graph_embedding_size=None):
        super(GraphPostEmbedding, self).__init__()

        if graph_embedding_size is None:
            graph_embedding_size = 2 * hidden_size

        self.node_gating_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid()
        )
        self.node_to_graph_net = torch.nn.Linear(hidden_size, graph_embedding_size)

    def forward(self, node_embedding, graph):
        scopes = graph_model.scopes_from_offsets(graph.node_offsets)

        transformed_embedding = self.node_gating_net(node_embedding) * self.node_to_graph_net(node_embedding)

        graph_embedding = sg_nn.functional.segment_avg_pool1d(
            transformed_embedding, scopes) * graph.node_counts.unsqueeze(-1)

        return graph_embedding
