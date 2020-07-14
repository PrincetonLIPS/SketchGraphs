import gzip
import json
import pickle

import numpy as np

import pytest

from sketchgraphs_models.graph import dataset as graph_dataset
from sketchgraphs_models.graph import model as graph_model

from sketchgraphs.data.sketch import Sketch

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


def test_compute_model(sketches, node_feature_mapping, edge_feature_mapping):
    sequences = list(map(graph_dataset.sketch_to_sequence, sketches))
    dataset = graph_dataset.GraphDataset(sequences, node_feature_mapping, edge_feature_mapping, seed=36)

    batch = [dataset[i] for i in range(10)]
    batch_input = graph_dataset.collate(batch, node_feature_mapping, edge_feature_mapping)

    model = graph_model.make_graph_model(32, {**node_feature_mapping.feature_dimensions, **edge_feature_mapping.feature_dimensions})
    result = model(batch_input)
    assert 'node_embedding' in result


def test_compute_losses(sketches, node_feature_mapping, edge_feature_mapping):
    sequences = list(map(graph_dataset.sketch_to_sequence, sketches))
    dataset = graph_dataset.GraphDataset(sequences, node_feature_mapping, edge_feature_mapping, seed=36)

    batch = [dataset[i] for i in range(10)]
    batch_input = graph_dataset.collate(batch, node_feature_mapping, edge_feature_mapping)

    feature_dimensions = {**node_feature_mapping.feature_dimensions, **edge_feature_mapping.feature_dimensions}

    model = graph_model.make_graph_model(32, feature_dimensions)
    model_output = model(batch_input)

    losses, _, edge_metrics, node_metrics = graph_model.compute_losses(
        model_output, batch_input, feature_dimensions)
    assert isinstance(losses, dict)

    avg_losses = graph_model.compute_average_losses(losses, batch_input['graph_counts'])
    assert isinstance(avg_losses, dict)

    for t in edge_metrics:
        assert edge_metrics[t][0].shape == edge_metrics[t][1].shape

    for t in edge_metrics:
        assert node_metrics[t][0].shape == node_metrics[t][1].shape


def test_compute_model_no_entity_features(sketches, edge_feature_mapping):
    sequences = list(map(graph_dataset.sketch_to_sequence, sketches))
    dataset = graph_dataset.GraphDataset(sequences, None, edge_feature_mapping, seed=36)

    batch = [dataset[i] for i in range(10)]
    batch_input = graph_dataset.collate(batch, None, edge_feature_mapping)

    model = graph_model.make_graph_model(
        32, {**edge_feature_mapping.feature_dimensions}, readout_entity_features=False)
    result = model(batch_input)
    assert 'node_embedding' in result


def test_compute_losses_no_entity_features(sketches, edge_feature_mapping):
    sequences = list(map(graph_dataset.sketch_to_sequence, sketches))
    dataset = graph_dataset.GraphDataset(sequences, None, edge_feature_mapping, seed=36)

    batch = [dataset[i] for i in range(10)]
    batch_input = graph_dataset.collate(batch, None, edge_feature_mapping)

    feature_dimensions = {**edge_feature_mapping.feature_dimensions}

    model = graph_model.make_graph_model(32, feature_dimensions, readout_entity_features=False)
    model_output = model(batch_input)

    losses, _, edge_metrics, node_metrics = graph_model.compute_losses(
        model_output, batch_input, feature_dimensions)
    assert isinstance(losses, dict)

    avg_losses = graph_model.compute_average_losses(losses, batch_input['graph_counts'])
    assert isinstance(avg_losses, dict)

    for t in edge_metrics:
        assert edge_metrics[t][0].shape == edge_metrics[t][1].shape

    for t in edge_metrics:
        assert node_metrics[t][0].shape == node_metrics[t][1].shape


def test_compute_model_subnode(sketches):
    sequences = list(map(graph_dataset.sketch_to_sequence, sketches))
    dataset = graph_dataset.GraphDataset(sequences, seed=36)

    batch = [dataset[i] for i in range(10)]
    batch_input = graph_dataset.collate(batch)

    model = graph_model.make_graph_model(
        32, feature_dimensions={}, readout_entity_features=False, readout_edge_features=False)
    result = model(batch_input)
    assert 'node_embedding' in result

    losses, _, edge_metrics, node_metrics = graph_model.compute_losses(
        result, batch_input, {})
    assert isinstance(losses, dict)

    avg_losses = graph_model.compute_average_losses(losses, batch_input['graph_counts'])
    assert isinstance(avg_losses, dict)
