"""Implementation of main graph-based model.

This module contains the implementation of the main graph-based model.

"""

import collections
import torch

from sketchgraphs_models import nn as sg_nn
from sketchgraphs_models.graph import dataset

from sketchgraphs.pipeline import graph_model
from sketchgraphs.pipeline.graph_model import target

from . import message_passing, numerical_features
from .losses import compute_losses, compute_average_losses


class GraphModel(torch.nn.Module):
    """Main model class for the graph-based models.

    Attributes
    ----------
    model_core : message_passing.GraphModelCore
        This module is the core of the network, and computes the main node and global embeddings
        in the graph.
    entity_label : torch.nn.Module
        This module computes the predictions for the label of the next node based on the global embedding.
    entity_feature_readout : Dict[TargetType, numerical_features.NumericalFeatureReadout], optional
        This dictionary contains feature readouts for each entity types, indexed by entity type.
    edge_post_embedding : torch.nn.Module
        This module computes an edge post embedding based one the node embedding of the two entities
        forming the edge.
    edge_label : torch.nn.Module
        This module computes the edge label from the embedding of the edge in question and the
        global embedding for the graph.
    edge_feature_readout : Dict[TartgetType, numerical_features.NumericalFeatureReadout], optional
        This dictionary contains feature readout for each edge type, indexed by the edge type.
    edge_partner : EdgePartnerNetwork
        This module computes the other vertex of a new added edge (on vertex is always the last vertex).
    """
    def __init__(self, model_core, entity_label, entity_feature_readout,
                 edge_post_embedding, edge_label, edge_feature_readout, edge_partner):
        super(GraphModel, self).__init__()
        self.model_core = model_core
        self.entity_label = entity_label
        self.entity_feature_readout = torch.nn.ModuleDict(entity_feature_readout)
        self.edge_post_embedding = edge_post_embedding
        self.edge_label = edge_label
        self.edge_partner = edge_partner
        self.edge_feature_readout = torch.nn.ModuleDict(edge_feature_readout)

    def forward(self, data):
        graph = data['graph']
        graph_counts = data['graph_counts']

        node_post_embedding, graph_embedding = self.model_core(data)

        with torch.autograd.profiler.record_function('edge_partner_readout'):
            partner_logits = self.edge_partner(node_post_embedding, graph_embedding, graph)

        total_constraint_counts = sum(graph_counts[t] for t in target.TargetType.edge_types())

        # Computation specific to constraint prediction
        with torch.autograd.profiler.record_function('edge_post_embedding'):
            current_node_post_embedding = node_post_embedding.index_select(
                0, graph.node_offsets[1:total_constraint_counts + 1] - 1)
            partner_node_post_embedding = node_post_embedding.index_select(0, data['edge_partner'])

            edge_post_embedding = self.edge_post_embedding(
                current_node_post_embedding, partner_node_post_embedding)

        with torch.autograd.profiler.record_function('edge_label_readout'):
            edge_label_logits = self.edge_label(edge_post_embedding, graph_embedding[:total_constraint_counts])

        graph_offsets = graph_model.offsets_from_counts(graph_counts)

        if not self.edge_feature_readout:
            edge_feature_logits = None
        else:
            with torch.autograd.profiler.record_function('edge_feature_readout'):
                edge_feature_logits = {
                    t: self.edge_feature_readout[t.name](
                        data['edge_numerical'][t],
                        torch.narrow(edge_post_embedding, 0, graph_offsets[t], graph_counts[t]),
                        torch.narrow(graph_embedding, 0, graph_offsets[t], graph_counts[t]))
                    for t in target.TargetType.numerical_edge_types()
                    if graph_counts[t] > 0
                }

        # Computation specific to entity type prediction
        total_entity_counts = sum(graph_counts[t] for t in target.TargetType.node_types())

        with torch.autograd.profiler.record_function('node_label_readout'):
            entity_logits = self.entity_label(
                graph_embedding[total_constraint_counts:total_constraint_counts+total_entity_counts])

        if not self.entity_feature_readout:
            node_feature_logits = None
        else:
            with torch.autograd.profiler.record_function('node_feature_readout'):
                node_feature_logits = {
                    t: self.entity_feature_readout[t.name](
                        data['node_numerical'][t],
                        torch.narrow(graph_embedding, 0, graph_offsets[t], graph_counts[t]))
                    for t in target.TargetType.numerical_node_types()
                    if graph_counts[t] > 0
                }

        result = {
            'graph_embedding': graph_embedding,
            'node_embedding': node_post_embedding,
            'partner_logits': partner_logits,
            'entity_logits': entity_logits,
            'edge_label_logits': edge_label_logits,
        }

        if edge_feature_logits is not None:
            result['edge_feature_logits'] = edge_feature_logits

        if node_feature_logits is not None:
            result['entity_feature_logits'] = node_feature_logits

        return result


class EdgePartnerNetwork(torch.nn.Module):
    """Predicts a probability for a new edge from a node to the last node in the graph."""
    def __init__(self, readout_net):
        super(EdgePartnerNetwork, self).__init__()

        self.readout_net = readout_net

    def forward(self, node_embedding, graph_embedding, graph):
        """Computes the logits at each node associated with the introducing
        the edge between that node and the target node in its associated graph.

        Note that the target specified by `target_idx[i]` is compared to its
        corresponding nodes in the ith graph.

        Parameters
        ----------
        node_embedding: tensor representing node embedding.
        graph_embedding: tensor representing graph embedding.
        graph: object describing the graph structure.

        Returns
        -------
        torch.Tensor
            The log-probability at each node for the corresponding edge
        """
        target_idx = graph.node_offsets[1:] - 1

        target_embeddings = (node_embedding
                             .index_select(0, target_idx)
                             .repeat_interleave(graph.node_counts, 0))

        graph_embedding = (graph_embedding
                           .repeat_interleave(graph.node_counts, 0))

        edge_partner_input = torch.cat((node_embedding, target_embeddings, graph_embedding), dim=-1)
        logits = self.readout_net(edge_partner_input).squeeze(-1)
        return logits


def make_graph_model(hidden_size, feature_dimensions, message_passing_rounds=3,
                     readout_edge_features=True, readout_entity_features=True,
                     readin_edge_features=None, readin_entity_features=None):
    """Create a graph model using the default inner architectural choices.

    The main graph model architecture is described as a large number of specific
    networks assembled together. This function chooses the architecture of all these
    networks from a couple of parameters to be specified.

    Parameters
    ----------
    hidden_size : int
        An integer parameter controlling the width of layers in the network.
    feature_dimensions : Dict[TargetType, List[int]]
        A dictionary describing the size of attribute features for each target type.
    message_passing_rounds : int
        An integer parameter controlling the number of message passing rounds in the model core.
    readout_edge_features : bool
        If true, indicates that the model should predict and output logits for edge features.
        Otherwise, edge feature prediction tasks are ignored.
    readout_entity_features : bool
        If true, indicates that the model should predict and output logits for entity features.
        Otherwise, entity feature prediction tasks are ignored.
    readin_edge_features : bool, optional
        If True, indicates that the model should use edge features as inputs. Otherwise, ignores
        input edge features (and only considers edge labels). If None, set to the same value as
        `readout_edge_features`. Note that setting this to a different value than `readout_edge_features`
        will prevent meaningful generation from the model.
    readin_entity_features : bool, optional
        If True, indicates that the model should use entity features as inputs. Otherwise, ignores
        input entity features (and only considers entity labels). If None, set to the same value
        as `readout_entity_features`.

    Returns
    -------
    GraphModel
        A new `GraphModel` instance with the default architecture.
    """
    if readin_edge_features is None:
        readin_edge_features = readout_edge_features

    if readin_entity_features is None:
        readin_entity_features = readout_entity_features

    # Build encoders and decoders for edge features
    if readout_edge_features or readin_edge_features:
        edge_feature_dimensions = collections.OrderedDict(
            (t, feature_dimensions[t]) for t in target.TargetType.numerical_edge_types())
        edge_embeddings, edge_readouts = numerical_features.make_embedding_and_readout(
            hidden_size, edge_feature_dimensions, numerical_features.edge_decoder_initial_input)
    else:
        edge_readouts = None

    if readin_edge_features:
        edge_embedding = message_passing.DenseSparsePreEmbedding(
            target.TargetType, edge_embeddings, len(target.EDGE_TYPES), hidden_size)
    else:
        edge_embedding = message_passing.DenseOnlyEmbedding(len(target.EDGE_TYPES), hidden_size)


    # Build encoders and decoders for node features
    if readout_entity_features or readin_entity_features:
        node_feature_dimensions = collections.OrderedDict(
            (t, feature_dimensions[t]) for t in target.TargetType.numerical_node_types())
        node_embeddings, node_readouts = numerical_features.make_embedding_and_readout(
            hidden_size, node_feature_dimensions, numerical_features.entity_decoder_initial_input)
    else:
        node_readouts = None

    if readin_entity_features:
        node_embedding = message_passing.DenseSparsePreEmbedding(
            target.TargetType, node_embeddings, len(target.NODE_TYPES), hidden_size)
    else:
        node_embedding = message_passing.DenseOnlyEmbedding(len(target.NODE_TYPES), hidden_size)

    # Build main model core
    model_core = message_passing.GraphModelCore(
        sg_nn.MessagePassingNetwork(
            message_passing_rounds,
            torch.nn.GRUCell(hidden_size, hidden_size),
            sg_nn.ConcatenateLinear(hidden_size, hidden_size, hidden_size)),
        node_embedding,
        edge_embedding,
        message_passing.GraphPostEmbedding(hidden_size, hidden_size),
    )

    return GraphModel(
        model_core,
        entity_label=sg_nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, len(target.NODE_TYPES_PREDICTED))
        ),
        entity_feature_readout=node_readouts,
        edge_post_embedding=sg_nn.ConcatenateLinear(hidden_size, hidden_size, hidden_size),
        edge_label=sg_nn.Sequential(
            sg_nn.ConcatenateLinear(hidden_size, hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, len(target.EDGE_TYPES_PREDICTED))
        ),
        edge_feature_readout=edge_readouts,
        edge_partner=EdgePartnerNetwork(
            torch.nn.Sequential(
                torch.nn.Linear(3 * hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, 1))))
