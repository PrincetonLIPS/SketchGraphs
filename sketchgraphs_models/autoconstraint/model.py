"""Main module which implements the components of graph-based autoconstraint model.
"""

import torch

from sketchgraphs.pipeline.graph_model import target, scopes_from_offsets

from sketchgraphs_models.graph.model import EdgePartnerNetwork, numerical_features, message_passing
from sketchgraphs_models import nn as sg_nn


class GlobalEmbeddingModelCore(torch.nn.Module):
    def __init__(self, embedding_dim, feature_dims, depth=3):
        super(GlobalEmbeddingModelCore, self).__init__()

        self.embedding_dim = embedding_dim

        self.node_embedding = message_passing.DenseSparsePreEmbedding(
            target.TargetType, {
                k.name: torch.nn.Sequential(
                    numerical_features.NumericalFeatureEncoding(fd.values(), embedding_dim),
                    numerical_features.NumericalFeaturesEmbedding(embedding_dim)
                )
                for k, fd in feature_dims.items()
            },
            len(target.NODE_TYPES), embedding_dim)

        self.global_entity_embedding = torch.nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=3)

        self.edge_embedding = torch.nn.Embedding(len(target.EDGE_TYPES), embedding_dim)

        self.message_passing = sg_nn.MessagePassingNetwork(
            depth, torch.nn.GRUCell(embedding_dim, embedding_dim),
            sg_nn.ConcatenateLinear(embedding_dim, embedding_dim, embedding_dim))

        self.graph_post_embedding = message_passing.GraphPostEmbedding(embedding_dim)

        self.merge_global_embedding = sg_nn.ConcatenateLinear(embedding_dim, 2 * embedding_dim, embedding_dim)

    def forward(self, data):
        graph = data['graph']

        with torch.autograd.profiler.record_function('entity_embeddings'):
            node_pre_embedding = self.node_embedding(data['node_features'].data, data['sparse_node_features'])

        with torch.autograd.profiler.record_function('global_embedding'):
            node_pre_embedding_packed = torch.nn.utils.rnn.PackedSequence(
                node_pre_embedding, data['node_features'].batch_sizes)
            output, _ = self.global_entity_embedding(node_pre_embedding_packed)

            global_embedding = output.data.index_select(0, data['last_graph_node_index'])

        with torch.autograd.profiler.record_function('message_passing'):
            node_pre_embedding_graph = node_pre_embedding.index_select(0, data['node_features_graph_index'])
            edge_pre_embedding = self.edge_embedding(graph.edge_features)

            node_post_embedding = self.message_passing(node_pre_embedding_graph, graph.incidence, (edge_pre_embedding,))
            graph_post_embedding = self.graph_post_embedding(node_post_embedding, graph)

        merged_global_embedding = self.merge_global_embedding(global_embedding, graph_post_embedding)

        return node_post_embedding, merged_global_embedding


class RecurrentEmbeddingModelCore(torch.nn.Module):
    def __init__(self, embedding_dim, feature_dims, depth=3):
        super(RecurrentEmbeddingModelCore, self).__init__()

        self.embedding_dim = embedding_dim

        self.node_embedding = message_passing.DenseSparsePreEmbedding(
            target.TargetType, {
                k.name: torch.nn.Sequential(
                    numerical_features.NumericalFeatureEncoding(fd.values(), embedding_dim),
                    numerical_features.NumericalFeaturesEmbedding(embedding_dim)
                )
                for k, fd in feature_dims.items()
            },
            len(target.NODE_TYPES), embedding_dim)

        self.global_entity_embedding = torch.nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=3)

        self.edge_embedding = torch.nn.Embedding(len(target.EDGE_TYPES), embedding_dim)

        self.message_passing = sg_nn.MessagePassingNetwork(
            depth, torch.nn.GRUCell(embedding_dim, embedding_dim),
            sg_nn.ConcatenateLinear(embedding_dim, embedding_dim, embedding_dim))

        self.graph_post_embedding = message_passing.GraphPostEmbedding(embedding_dim)

        self.merge_global_embedding = sg_nn.ConcatenateLinear(embedding_dim, 2 * embedding_dim, embedding_dim)


    def forward(self, data):
        graph = data['graph']

        with torch.autograd.profiler.record_function('entity_embeddings'):
            node_pre_embedding = self.node_embedding(data['node_features'].data, data['sparse_node_features'])

        with torch.autograd.profiler.record_function('global_embedding'):
            node_pre_embedding_packed = torch.nn.utils.rnn.PackedSequence(
                node_pre_embedding, data['node_features'].batch_sizes)
            node_pre_embedding_transformed, state = self.global_entity_embedding(node_pre_embedding_packed)

            global_embedding = state[-1]

        with torch.autograd.profiler.record_function('message_passing'):
            node_pre_embedding_graph = node_pre_embedding_transformed.data.index_select(0, data['node_features_graph_index'])
            edge_pre_embedding = self.edge_embedding(graph.edge_features)

            node_post_embedding = self.message_passing(node_pre_embedding_graph, graph.incidence, (edge_pre_embedding,))
            graph_post_embedding = self.graph_post_embedding(node_post_embedding, graph)

        merged_global_embedding = self.merge_global_embedding(global_embedding, graph_post_embedding)

        return node_post_embedding, merged_global_embedding


class BidirectionalRecurrentModelCore(torch.nn.Module):
    def __init__(self, embedding_dim, feature_dims, depth=3):
        super(BidirectionalRecurrentModelCore, self).__init__()

        self.embedding_dim = embedding_dim

        self.node_embedding = message_passing.DenseSparsePreEmbedding(
            target.TargetType, {
                k.name: torch.nn.Sequential(
                    numerical_features.NumericalFeatureEncoding(fd.values(), embedding_dim),
                    numerical_features.NumericalFeaturesEmbedding(embedding_dim)
                )
                for k, fd in feature_dims.items()
            },
            len(target.NODE_TYPES), embedding_dim)

        self.node_pre_embedding_transform = torch.nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=3,
            bidirectional=True)

        self.node_pre_embedding_merge_direction = torch.nn.Linear(2 * embedding_dim, embedding_dim)

        self.edge_embedding = torch.nn.Embedding(len(target.EDGE_TYPES), embedding_dim)

        self.message_passing = sg_nn.MessagePassingNetwork(
            depth, torch.nn.GRUCell(embedding_dim, embedding_dim),
            sg_nn.ConcatenateLinear(embedding_dim, embedding_dim, embedding_dim))

        self.graph_post_embedding = message_passing.GraphPostEmbedding(embedding_dim)

        self.merge_global_embedding = sg_nn.ConcatenateLinear(2 * embedding_dim, 2 * embedding_dim, embedding_dim)

    def forward(self, data):
        graph = data['graph']

        with torch.autograd.profiler.record_function('entity_embeddings'):
            node_pre_embedding = self.node_embedding(data['node_features'].data, data['sparse_node_features'])

        with torch.autograd.profiler.record_function('global_embedding'):
            node_pre_embedding_packed = torch.nn.utils.rnn.PackedSequence(
                node_pre_embedding, data['node_features'].batch_sizes)
            node_pre_embedding_transformed, state = self.node_pre_embedding_transform(node_pre_embedding_packed)

            global_embedding = torch.flatten(torch.transpose(
                state.view(3, 2, -1, self.embedding_dim)[-1], 0, 1),
                start_dim=1)

        with torch.autograd.profiler.record_function('message_passing'):
            node_pre_embedding_graph_bidir = node_pre_embedding_transformed.data.index_select(0, data['node_features_graph_index'])
            node_pre_embedding_graph = self.node_pre_embedding_merge_direction(node_pre_embedding_graph_bidir)
            edge_pre_embedding = self.edge_embedding(graph.edge_features)

            node_post_embedding = self.message_passing(node_pre_embedding_graph, graph.incidence, (edge_pre_embedding,))
            graph_post_embedding = self.graph_post_embedding(node_post_embedding, graph)

        merged_global_embedding = self.merge_global_embedding(global_embedding, graph_post_embedding)

        return node_post_embedding, merged_global_embedding


MODEL_CORES = {
    'global_embedding': GlobalEmbeddingModelCore,
    'recurrent_embedding': RecurrentEmbeddingModelCore,
    'bidirectional_recurrent': BidirectionalRecurrentModelCore
}


class AutoconstraintModel(torch.nn.Module):
    def __init__(self, model_core):
        super(AutoconstraintModel, self).__init__()

        self.model_core = model_core

        embedding_dim = model_core.embedding_dim

        self.edge_partner_network = EdgePartnerNetwork(
            torch.nn.Sequential(
                torch.nn.Linear(3 * embedding_dim, embedding_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(embedding_dim, 1)))

        self.edge_label = torch.nn.Sequential(
            torch.nn.Linear(3 * embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, len(target.EDGE_TYPES_PREDICTED)))


    def _compute_label_logits_partner(self, graph, node_post_embedding, global_embedding, partner_index):
        node_current_post_embedding_label = node_post_embedding.index_select(
            0, graph.node_offsets[1:][partner_index.index] - 1)
        node_partner_post_embedding_label = node_post_embedding.index_select(0, partner_index.values)
        merged_global_embedding_label = global_embedding.index_select(0, partner_index.index)

        edge_label_input = torch.cat(
            (node_current_post_embedding_label, node_partner_post_embedding_label, merged_global_embedding_label), dim=-1)

        return self.edge_label(edge_label_input)

    def _compute_all_label_logits(self, graph, node_post_embedding, global_embedding):
        node_current_post_embedding = (node_post_embedding
            .index_select(0, graph.node_offsets[1:] - 1)
            .repeat_interleave(graph.node_counts, 0))

        global_embedding = global_embedding.repeat_interleave(graph.node_counts, 0)

        edge_label_input = torch.cat(
            (node_current_post_embedding, node_post_embedding, global_embedding), dim=-1)

        return self.edge_label(edge_label_input)

    def forward(self, data, compute_all_label_logits=False):
        graph = data['graph']

        node_post_embedding, merged_global_embedding = self.model_core(data)

        with torch.autograd.profiler.record_function('edge_partner'):
            edge_partner_logits = self.edge_partner_network(
                node_post_embedding, merged_global_embedding, graph)

        with torch.autograd.profiler.record_function('edge_label'):
            if compute_all_label_logits:
                edge_label_logits = self._compute_all_label_logits(
                    graph, node_post_embedding, merged_global_embedding)
            else:
                edge_label_logits = self._compute_label_logits_partner(
                    graph, node_post_embedding, merged_global_embedding, data['partner_index'])

        return {
            'edge_partner_logits': edge_partner_logits,
            'edge_label_logits': edge_label_logits
        }


def segment_stop_loss(partner_logits, segment_offsets, partner_index, stop_partner_index_index, reduction='sum'):
    scopes = scopes_from_offsets(segment_offsets)

    log_weight_other = sg_nn.functional.segment_logsumexp(partner_logits, scopes)
    total_weight = torch.nn.functional.softplus(log_weight_other)

    stop_loss = total_weight.index_select(0, stop_partner_index_index)
    partner_loss = (total_weight.index_select(0, partner_index.index)
                    - partner_logits.index_select(0, partner_index.value))

    if reduction == 'sum':
        return stop_loss.sum(), partner_loss.sum()
    elif reduction == 'mean':
        return stop_loss.mean(), partner_loss.mean()
    elif reduction == 'none':
        return stop_loss, partner_loss
    else:
        raise ValueError('Reduction must be one of sum, mean or none.')


def segment_stop_accuracy(partner_logits, segment_offsets, target_idx, stop_partner_index_index):
    """Computes the accuracy for stop prediction for partner logits."""
    scopes = scopes_from_offsets(segment_offsets)

    max_logit_in_segment, max_logit_indices = sg_nn.functional.segment_argmax(
        partner_logits.detach(), scopes)

    prediction_stop_correct = (max_logit_in_segment < 0).index_select(0, stop_partner_index_index)

    prediction_partner_correct = (
        (max_logit_in_segment > 0).index_select(0, target_idx.indices) &
        ((max_logit_indices + segment_offsets[:-1]).index_select(0, target_idx.indices) == target_idx.values))

    return prediction_stop_correct.float().mean(), prediction_partner_correct.float().mean()


def compute_losses(data, readout, reduction='sum'):
    edge_stop_loss, partner_loss = segment_stop_loss(
        readout['edge_partner_logits'], data['graph'].node_offsets,
        data['partner_index'], data['stop_partner_index_index'],
        reduction=reduction)

    edge_stop_accuracy, edge_partner_accuracy = segment_stop_accuracy(
        readout['edge_partner_logits'], data['graph'].node_offsets,
        data['partner_index'], data['stop_partner_index_index'])

    edge_label_loss = torch.nn.functional.cross_entropy(
        readout['edge_label_logits'], data['edge_label'],
        reduction=reduction)

    if len(readout['edge_label_logits']) > 0:
        edge_label_accuracy = (torch.argmax(readout['edge_label_logits'], dim=-1) == data['edge_label']).float().mean()
    else:
        edge_label_accuracy = readout['edge_label_logits'].new_empty([0])

    return {
        'edge_partner': partner_loss,
        'edge_stop': edge_stop_loss,
        'edge_label': edge_label_loss,
    }, {
        'edge_partner': edge_partner_accuracy,
        'edge_stop': edge_stop_accuracy,
        'edge_label': edge_label_accuracy,
    }


def compute_average_losses(data, losses):
    result = {}
    result['edge_partner'] = losses['edge_partner'] / data['edge_label'].shape[0]
    result['edge_label'] = losses['edge_label'] / data['edge_label'].shape[0]
    result['edge_stop'] = losses['edge_stop'] / data['stop_partner_index_index'].shape[0]
    return result
