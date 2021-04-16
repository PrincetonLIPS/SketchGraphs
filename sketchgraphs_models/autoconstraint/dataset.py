"""Dataset for auto-constraint model."""

from typing import Optional

import numpy as np
import torch

from sketchgraphs.data import sequence as datalib
from sketchgraphs.pipeline.graph_model.target import NODE_TYPES, EDGE_TYPES, EDGE_TYPES_PREDICTED, NODE_IDX_MAP, EDGE_IDX_MAP
from sketchgraphs.pipeline import graph_model as graph_utils

from sketchgraphs_models.graph.dataset import EntityFeatureMapping, EdgeFeatureMapping, _sparse_feature_to_torch


def _reindex_sparse_batch(sparse_batch, pack_batch_offsets):
    return graph_utils.SparseFeatureBatch(
        pack_batch_offsets[sparse_batch.index],
        sparse_batch.value)


def collate(batch):
    # Sort batch for packing
    node_lengths = [len(x['node_features']) for x in batch]
    sorted_indices = np.argsort(node_lengths)[::-1].copy()

    batch = [batch[i] for i in sorted_indices]

    graph = graph_utils.GraphInfo.merge(*[x['graph'] for x in batch])
    edge_label = torch.tensor(
        [x['target_edge_label'] for x in batch if x['target_edge_label'] != -1], dtype=torch.int64)
    node_features = torch.nn.utils.rnn.pack_sequence([x['node_features'] for x in batch])
    batch_offsets = graph_utils.offsets_from_counts(node_features.batch_sizes)

    node_features_graph_index = torch.cat([
        i + batch_offsets[:graph.node_counts[i]] for i in range(len(batch))
    ], dim=0)

    sparse_node_features = {}

    for k in batch[0]['sparse_node_features']:
        sparse_node_features[k] = graph_utils.SparseFeatureBatch.merge(
            [_reindex_sparse_batch(x['sparse_node_features'][k], batch_offsets) for x in batch], range(len(batch)))

    last_graph_node_index = batch_offsets[graph.node_counts - 1] + torch.arange(len(graph.node_counts), dtype=torch.int64)

    partner_index_index = []
    partner_index = []

    stop_partner_index_index = []

    for i, x in enumerate(batch):
        if x['partner_index'] == -1:
            stop_partner_index_index.append(i)
            continue

        partner_index_index.append(i)
        partner_index.append(x['partner_index'] + graph.node_offsets[i])

    partner_index = graph_utils.SparseFeatureBatch(
        torch.tensor(partner_index_index, dtype=torch.int64),
        torch.tensor(partner_index, dtype=torch.int64)
    )

    stop_partner_index_index = torch.tensor(stop_partner_index_index, dtype=torch.int64)

    return {
        'graph': graph,
        'edge_label': edge_label,
        'partner_index': partner_index,
        'stop_partner_index_index': stop_partner_index_index,
        'node_features': node_features,
        'node_features_graph_index': node_features_graph_index,
        'sparse_node_features': sparse_node_features,
        'last_graph_node_index': last_graph_node_index,
        'sorted_indices': torch.as_tensor(sorted_indices)
    }



def process_node_and_edge_ops(node_ops, edge_ops_in_graph, num_nodes_in_graph, node_feature_mappings: Optional[EntityFeatureMapping]):
    all_node_labels = torch.tensor([NODE_IDX_MAP[op.label] for op in node_ops], dtype=torch.int64)
    edge_labels = torch.tensor([EDGE_IDX_MAP[op.label] for op in edge_ops_in_graph], dtype=torch.int64)

    if len(edge_ops_in_graph) > 0:
        incidence = torch.tensor([(op.references[0], op.references[-1]) for op in edge_ops_in_graph],
                                 dtype=torch.int64).T.contiguous()
        incidence = torch.cat((incidence, torch.flip(incidence, [0])), dim=1)
    else:
        incidence = torch.empty([2, 0], dtype=torch.int64)

    edge_features = edge_labels.repeat(2)

    if node_feature_mappings is not None:
        sparse_node_features = _sparse_feature_to_torch(node_feature_mappings.all_sparse_features(node_ops))
    else:
        sparse_node_features = None

    graph = graph_utils.GraphInfo.from_single_graph(incidence, None, edge_features, num_nodes_in_graph)

    return {
        'graph': graph,
        'node_features': all_node_labels,
        'sparse_node_features': sparse_node_features
    }



class AutoconstraintDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, node_feature_mappings, seed=10):
        self.sequences = sequences
        self.node_feature_mappings = node_feature_mappings
        self._rng = np.random.Generator(np.random.Philox(seed))

    def __getitem__(self, idx):
        idx = idx % len(self.sequences)
        seq = self.sequences[idx]

        if not isinstance(seq[0], datalib.NodeOp):
            raise ValueError('First operation in sequence is not a NodeOp')

        if seq[-1].label != datalib.EntityType.Stop:
            seq.append(datalib.NodeOp(datalib.EntityType.Stop, {}))

        node_ops = [seq[0]]
        edge_ops = []

        num_predicted_edge_ops_per_node = []
        num_non_predicted_edge_ops_per_node = []

        predicted_edge_ops_for_current_node = 0
        non_predicted_edge_ops_for_current_node = 0

        for op in seq[1:]:
            if isinstance(op, datalib.NodeOp):
                num_predicted_edge_ops_per_node.append(predicted_edge_ops_for_current_node)
                num_non_predicted_edge_ops_per_node.append(non_predicted_edge_ops_for_current_node)

                predicted_edge_ops_for_current_node = 0
                non_predicted_edge_ops_for_current_node = 0

                node_ops.append(op)
            else:
                if op.label in EDGE_TYPES_PREDICTED:
                    predicted_edge_ops_for_current_node += 1
                else:
                    non_predicted_edge_ops_for_current_node += 1

                edge_ops.append(op)

        node_ops = node_ops[:-1]

        num_predicted_edge_ops_per_node = np.array(num_predicted_edge_ops_per_node, dtype=np.int64)
        num_non_predicted_edge_ops_per_node = np.array(num_non_predicted_edge_ops_per_node, dtype=np.int64)

        predicted_edge_ops_offsets = num_predicted_edge_ops_per_node.cumsum()
        non_predicted_edge_ops_offsets = num_non_predicted_edge_ops_per_node.cumsum()

        num_predicted_edge_ops = predicted_edge_ops_offsets[-1]

        stop_target = self._rng.uniform() < len(node_ops) / (len(node_ops) + num_predicted_edge_ops)

        if stop_target:
            target_node_idx = self._rng.integers(len(node_ops))
            num_nodes_in_graph = target_node_idx + 1
            edge_ops_in_graph = edge_ops[:predicted_edge_ops_offsets[target_node_idx] + non_predicted_edge_ops_offsets[target_node_idx]]
            target_edge_label = -1
            partner_index = -1
        else:
            target_predicted_edge_idx = self._rng.integers(num_predicted_edge_ops)
            target_node_idx = np.searchsorted(predicted_edge_ops_offsets, target_predicted_edge_idx, side='right')
            num_nodes_in_graph = target_node_idx + 1

            target_edge_idx = target_predicted_edge_idx + non_predicted_edge_ops_offsets[target_node_idx]
            target_edge = edge_ops[target_edge_idx]
            edge_ops_in_graph = edge_ops[:target_edge_idx]
            target_edge_label = EDGE_IDX_MAP[target_edge.label]
            partner_index = target_edge.references[-1]
            assert target_edge_label < len(EDGE_TYPES_PREDICTED)

        input_features = process_node_and_edge_ops(
            node_ops, edge_ops_in_graph, num_nodes_in_graph, self.node_feature_mappings)

        return {
            **input_features,
            'target_edge_label': target_edge_label,
            'partner_index': partner_index,
        }

__all__ = [
    'NODE_TYPES', 'EDGE_TYPES', 'EDGE_TYPES_PREDICTED', 'NODE_IDX_MAP', 'EDGE_IDX_MAP',
    'EntityFeatureMapping', 'EdgeFeatureMapping', 'collate', 'AutoconstraintDataset'
]
