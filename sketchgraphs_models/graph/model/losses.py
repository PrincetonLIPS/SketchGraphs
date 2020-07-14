"""This module implements the necessary code to compute losses from the output of the graph model.
"""

from typing import Any, Dict

import torch

from sketchgraphs.pipeline.graph_model import target, scopes_from_offsets
from sketchgraphs_models.nn import functional as sg_functional

from .. import dataset



def segment_stop_loss(partner_logits, segment_offsets, target_idx=None):
    """Computes the loss for stop prediction for partner logits.

    This function effectively corresponds to a cross-entropy softmax
    operation on each segment, where the cross-entropy is augmented
    with one last constant logit. If the target predicted is omitted,
    we assume that it is the last (implicit) entry.

    Parameters
    ----------
    partner_logits : torch.Tensor
        The un-normalized logits for the segments.
    segment_offsets : torch.Tensor
        A tensor of shape `[num_segments + 1]` indicating the segment offsets.
    target_idx : torch.Tensor, optional
        If not None, a tensor of shape `[num_segments]` representing the target offset to predict in each segment.
        Otherwise, this is assumed to be the last (implicit) entry in each segment.

    Returns
    -------
    torch.Tensor
        A tensor of shape `[num_segments]` representing the cross-entropy loss at each segment.
    """
    scopes = scopes_from_offsets(segment_offsets)

    log_weight_other = sg_functional.segment_logsumexp(partner_logits, scopes)
    total_weight = torch.nn.functional.softplus(log_weight_other)

    if target_idx is None:
        return total_weight
    else:
        return total_weight - partner_logits.index_select(0, target_idx)


def segment_stop_accuracy(partner_logits, segment_offsets, target_idx=None):
    """Computes the accuracy for stop prediction for partner logits.

    Parameters
    ----------
    partner_logits : torch.Tensor
        The un-normalized logits for the segments.
    segment_offsets : torch.Tensor
        A tensor of shape `[num_segments + 1]` indicating the segment offsets.
    target_idx : torch.Tensor
        If not None, a tensor of shape `[num_segments]` representing the target offset to predict in each segment.
        Otherwise, this is assumed to be the last (implicit) entry in each segment.

    Returns
    -------
    torch.Tensor
        A boolean tensor of shape `[num_segments]` representing the accuracy at each segment.
    """
    scopes = scopes_from_offsets(segment_offsets)

    max_logit_in_segment, max_logit_indices = sg_functional.segment_argmax(
        partner_logits.detach(), scopes)

    if target_idx is None:
        return max_logit_in_segment < 0
    else:
        return (max_logit_in_segment > 0) & (max_logit_indices + segment_offsets[:-1] == target_idx)


def compute_edge_losses(edge_partner_offsets, edge_label_logits, partner_logits, edge_label, edge_partner):
    """Computes losses associated with edge prediction problems.

    This function computes the two losses associated with the edge problem:
    - `edge_label`: this loss represents the loss for predicting the label of the edge
    - `edge_partner`: this loss represents the loss for predicting the other vertex of the edge
    """
    edge_label_loss = torch.nn.functional.cross_entropy(edge_label_logits, edge_label, reduction='sum')
    edge_label_accuracy = (torch.argmax(edge_label_logits, dim=-1) == edge_label).float().mean()

    edge_partner_logits = partner_logits[:edge_partner_offsets[-1]]

    edge_partner_loss = segment_stop_loss(
        edge_partner_logits, edge_partner_offsets, target_idx=edge_partner).sum()
    edge_partner_accuracy = segment_stop_accuracy(
        edge_partner_logits, edge_partner_offsets, target_idx=edge_partner).float().mean()

    return {
        'edge_label': edge_label_loss,
        'edge_partner': edge_partner_loss
    }, {
        'edge_label': edge_label_accuracy,
        'edge_partner': edge_partner_accuracy
    }


def compute_node_losses(node_offsets, entity_logits, partner_logits, node_label):
    node_label_loss = torch.nn.functional.cross_entropy(entity_logits, node_label, reduction='sum')
    node_label_accuracy = (torch.argmax(entity_logits, dim=-1) == node_label).float().mean()


    node_partner_logits = partner_logits[node_offsets[0]:node_offsets[-1]]
    node_partner_offsets = node_offsets - node_offsets[0]

    node_stop_loss = segment_stop_loss(
        node_partner_logits, node_partner_offsets).sum()

    node_stop_accuracy = segment_stop_accuracy(
        node_partner_logits, node_partner_offsets).float().mean()

    return {
        'node_label': node_label_loss,
        'node_stop': node_stop_loss
    }, {
        'node_label': node_label_accuracy,
        'node_stop': node_stop_accuracy
    }


def compute_subnode_losses(subnode_offsets, partner_logits):
    subnode_partner_logits = partner_logits[subnode_offsets[0]:subnode_offsets[-1]]
    subnode_partner_offsets = subnode_offsets - subnode_offsets[0]

    subnode_stop_loss = segment_stop_loss(
        subnode_partner_logits, subnode_partner_offsets).sum()

    subnode_stop_accuracy = segment_stop_accuracy(
        subnode_partner_logits, subnode_partner_offsets).float().mean()

    return {
        'subnode_stop': subnode_stop_loss
    }, {
        'subnode_stop': subnode_stop_accuracy
    }


def merge_losses_and_accuracy(losses, accuracy, updates, weight=None):
    if weight is not None:
        losses = {k: l * weight for k, l in losses.items()}

    losses.update(updates[0])
    accuracy.update(updates[1])


def compute_feature_loss(feature_logits: torch.Tensor, feature_targets: torch.Tensor,
                         feature_dimensions: Dict[Any, int]):
    """Computes losses on the given features.

    Parameters
    ----------
    feature_logits : torch.Tensor
        2-d tensor of feature logits
    feature_targets : torch.Tensor
        1-d integer tensor of true feature labels
    feature_dimensions : Dict[str, int]
        A list of integers representing the dimension of each feature.

    Returns
    -------
    losses : torch.Tensor
        An array of losses on each feature
    accuracies : torch.Tensor
        An array of accuracies on each feature
    labels : torch.Tensor
        The provided array of targets
    predictions : torch.Tensor
        An array of predicted labels according to arg-max predictions.
    """
    current_offset = 0

    losses = feature_logits.new_empty(len(feature_dimensions))
    accuracies = feature_logits.new_empty(len(feature_dimensions))

    predictions = feature_logits.new_empty((feature_logits.shape[0], len(feature_dimensions)), dtype=torch.int64)

    for i, feature_dim in enumerate(feature_dimensions.values()):
        logits = torch.narrow(feature_logits, 1, current_offset, feature_dim)
        prediction = torch.argmax(logits, dim=-1)

        accuracies[i] = (prediction == feature_targets[:, i]).float().mean()
        losses[i] = torch.nn.functional.cross_entropy(logits, feature_targets[:, i], reduction='sum')
        predictions[:, i] = prediction

        current_offset += feature_dim

    labels = feature_targets

    return losses, accuracies, labels, predictions


def compute_losses(readout, batch, feature_dimensions, weights=None):
    """Computes losses for each component of the model, given the model output.

    Note that this function computes losses using a sum reduction.
    To obtain equivalent average losses, see `compute_average_losses`.

    This function also returns labels and predictions for numerical edges' numerical features.

    Parameters
    ----------
    readout : dict
        A dictionary containing the model output
    batch : dict
        A dictionary containing the input data
    feature_dimensions : dict
        A dictionary of list of ints describing the dimension of each feature for each target.
    weights : dict, optional
        If not None, a mapping from TargetType to floats which describes the weighting of each prediction
        endpoint. TargetTypes which are not included are assumed to be weighted at 1.

    Returns
    -------
    losses : dict
        Nested dictionary containing loss values for each component
    accuracy : dict
        Nested dictionary containing accuracy for each component
    efeat_labels: dict
        Dictionary containing labels for each numerical edge feature.
    efeat_preds: dict
        Dictionary containing predictions for each numerical edge feature.
    """
    if weights is None:
        weights = {}

    graph = batch['graph']
    counts = batch['graph_counts']


    count_edge = sum(counts[t] for t in target.TargetType.edge_types())

    losses = {'edge_features': {}, 'node_features': {}}
    accuracy = {'edge_features': {}, 'node_features': {}}
    edge_metrics = {}
    node_metrics = {}

    if count_edge > 0:
        with torch.autograd.profiler.record_function('edge_loss'):
            edge_partner_offsets = graph.node_offsets[:count_edge + 1]
            merge_losses_and_accuracy(
                losses, accuracy,
                compute_edge_losses(
                    edge_partner_offsets, readout['edge_label_logits'], readout['partner_logits'],
                    batch['edge_label'], batch['edge_partner']),
                weights.get(target.TargetType.EdgeCategorical))

        if 'edge_feature_logits' in readout:
            with torch.autograd.profiler.record_function('edge_feature_losses'):
                for t, feature_logit in readout['edge_feature_logits'].items():
                    weight = weights.get(t)
                    losses['edge_features'][t], accuracy['edge_features'][t], edge_label, edge_prob = compute_feature_loss(
                        feature_logit, batch['edge_numerical'][t], feature_dimensions[t])
                    if weight is not None:
                        losses['edge_features'][t] *= weight
                    edge_metrics[t] = (edge_label, edge_prob)

    count_nodes = sum(counts[t] for t in target.TargetType.node_types())

    if count_nodes > 0:
        with torch.autograd.profiler.record_function('entity_loss'):
            node_graph_offsets = graph.node_offsets.narrow(0, count_edge, count_nodes + 1)
            merge_losses_and_accuracy(
                losses, accuracy,
                compute_node_losses(
                    node_graph_offsets, readout['entity_logits'], readout['partner_logits'], batch['node_label']),
                weights.get(target.TargetType.NodeGeneric))

        if 'entity_feature_logits' in readout:
            with torch.autograd.profiler.record_function('entity_feature_loss'):
                for t, feature_logit in readout['entity_feature_logits'].items():
                    weight = weights.get(t)
                    losses['node_features'][t], accuracy['node_features'][t], node_label, node_prob = compute_feature_loss(
                        feature_logit, batch['node_numerical'][t], feature_dimensions[t])
                    if weight is not None:
                        losses['node_features'][t] *= weight
                    node_metrics[t] = (node_label, node_prob)

    if counts[target.TargetType.Subnode] > 0:
        with torch.autograd.profiler.record_function('subnode_loss'):
            subnode_graphs_offsets = graph.node_offsets[count_edge + count_nodes:]
            merge_losses_and_accuracy(
                losses, accuracy,
                compute_subnode_losses(
                    subnode_graphs_offsets, readout['partner_logits']),
                weights.get(target.TargetType.Subnode))

    return losses, accuracy, edge_metrics, node_metrics


_loss_to_target_type = {
    'edge_label': target.TargetType.edge_types(),
    'edge_partner': target.TargetType.edge_types(),
    'node_stop': target.TargetType.node_types(),
    'node_label': target.TargetType.node_types(),
    'subnode_stop': [target.TargetType.Subnode],
}

def compute_average_losses(losses, graph_counts):
    """Computes average losses from sum losses.

    Parameters
    ----------
    losses : dict
        Dictionary containing components for each loss
    graph_counts : Mapping
        Mapping from target types to integers representing the number of aggregated examples
        for each target type.

    Returns
    -------
    dict
        A dictionary containing the average loss values
    """
    avg_losses = {}

    for k, v in losses.items():
        if k == 'edge_features' or k == 'node_features':
            continue

        avg_losses[k] = v / float(sum(graph_counts[t] for t in _loss_to_target_type[k]))

    if 'edge_features' in losses:
        avg_edge_feature_losses = {}
        for k, v in losses['edge_features'].items():
            if graph_counts[k] > 0:
                avg_edge_feature_losses[k] = v / float(graph_counts[k])
    else:
        avg_edge_feature_losses = None

    if 'node_features' in losses:
        avg_node_feature_losses = {}
        for k, v in losses['node_features'].items():
            if graph_counts[k] > 0:
                avg_node_feature_losses[k] = v / float(graph_counts[k])
    else:
        avg_node_feature_losses = None

    if avg_edge_feature_losses is not None:
        avg_losses['edge_features'] = avg_edge_feature_losses
    if avg_node_feature_losses is not None:
        avg_losses['node_features'] = avg_node_feature_losses

    return avg_losses
