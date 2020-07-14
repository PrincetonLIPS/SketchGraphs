import gzip
import json
import pickle
import pytest

import numpy as np

from sketchgraphs_models.graph import dataset as graph_dataset
from sketchgraphs.data import sequence as data_sequence

@pytest.fixture
def edge_feature_mapping():
    """Dummy quantization functions."""
    with gzip.open('tests/testdata/sg_t16.stats.pkl.gz', 'rb') as f:
        mapping = pickle.load(f)

    return graph_dataset.EdgeFeatureMapping(
        mapping['edge']['angle'], mapping['edge']['length'])

@pytest.fixture
def node_feature_mapping():
    with gzip.open('tests/testdata/sg_t16.stats.pkl.gz', 'rb') as f:
        mapping = pickle.load(f)
    return graph_dataset.EntityFeatureMapping(mapping['node'])


def test_dataset(sketches, node_feature_mapping, edge_feature_mapping):
    sequences = list(map(data_sequence.sketch_to_sequence, sketches))
    dataset = graph_dataset.GraphDataset(sequences, node_feature_mapping, edge_feature_mapping, seed=12)

    graph, target = dataset[0]
    assert graph_dataset.TargetType.EdgeDistance in graph.sparse_edge_features
    assert graph.node_features is not None
    assert graph.sparse_node_features is not None


def test_collate_empty(node_feature_mapping, edge_feature_mapping):
    result = graph_dataset.collate([], node_feature_mapping, edge_feature_mapping)
    assert 'edge_numerical' in result


def test_collate_some(sketches, node_feature_mapping, edge_feature_mapping):
    sequences = list(map(data_sequence.sketch_to_sequence, sketches))
    dataset = graph_dataset.GraphDataset(sequences, node_feature_mapping, edge_feature_mapping, seed=42)

    batch = [dataset[i] for i in range(5)]

    batch_info = graph_dataset.collate(batch, node_feature_mapping, edge_feature_mapping)
    assert 'edge_numerical' in batch_info
    assert batch_info['graph'].node_features is not None
