"""This module contains the main functions used to load the required data from disk for training."""

import functools
import gzip
import pickle
import os

import numpy as np
import torch

from sketchgraphs_models import distributed_utils
from sketchgraphs_models.nn import data_util
from sketchgraphs_models.graph import dataset

from sketchgraphs.data import flat_array


def load_sequences_and_mappings(dataset_file, auxiliary_file, quantization, entity_features=True, edge_features=True):
    data = flat_array.load_dictionary_flat(np.load(dataset_file, mmap_mode='r'))

    if auxiliary_file is None:
        root, _ = os.path.splitext(dataset_file)
        auxiliary_file = root + '.stats.pkl.gz'

    if entity_features or edge_features:
        with gzip.open(auxiliary_file, 'rb') as f:
            auxiliary_dict = pickle.load(f)

    if entity_features:
        entity_feature_mapping = dataset.EntityFeatureMapping(auxiliary_dict['node'])
    else:
        entity_feature_mapping = None

    seqs = data['sequences']
    weights = data['sequence_lengths']

    if edge_features:
        if isinstance(quantization['angle'], dataset.QuantizationMap):
            angle_map = quantization['angle']
        else:
            angle_map = dataset.QuantizationMap.from_counter(auxiliary_dict['edge']['angle'], quantization['angle'])

        if isinstance(quantization['length'], dataset.QuantizationMap):
            length_map = quantization['length']
        else:
            length_map = dataset.QuantizationMap.from_counter(auxiliary_dict['edge']['length'], quantization['length'])
        edge_feature_mapping = dataset.EdgeFeatureMapping(angle_map, length_map)
    else:
        edge_feature_mapping = None

    return {
        'sequences': seqs.share_memory_(),
        'entity_feature_mapping': entity_feature_mapping,
        'edge_feature_mapping': edge_feature_mapping,
        'weights': weights
    }


def load_dataset_and_weights_with_mapping(dataset_file, node_feature_mapping, edge_feature_mapping, seed=None):
    data = flat_array.load_dictionary_flat(np.load(dataset_file, mmap_mode='r'))
    seqs = data['sequences']
    seqs.share_memory_()

    ds = dataset.GraphDataset(seqs, node_feature_mapping, edge_feature_mapping, seed)

    return ds, data['sequence_lengths']


def load_dataset_and_weights(dataset_file, auxiliary_file, quantization, seed=None,
                             entity_features=True, edge_features=True, force_entity_categorical_features=False):
    data = load_sequences_and_mappings(dataset_file, auxiliary_file, quantization, entity_features, edge_features)

    if data['entity_feature_mapping'] is None and force_entity_categorical_features:
        # Create an entity mapping which only computes the categorical features (i.e. isConstruction and clockwise)
        data['entity_feature_mapping'] = dataset.EntityFeatureMapping()

    return dataset.GraphDataset(
        data['sequences'], data['entity_feature_mapping'], data['edge_feature_mapping'], seed=seed), data['weights']


def make_dataloader_train(collate_fn, ds_train, weights, batch_size, num_epochs, num_workers, distributed_config=None):
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, len(weights), replacement=True)

    if distributed_config is not None:
        sampler = distributed_utils.DistributedSampler(
            sampler, distributed_config.world_size, distributed_config.rank)

    batch_sampler = torch.utils.data.BatchSampler(
        sampler, batch_size, drop_last=False)

    dataloader_train = torch.utils.data.DataLoader(
        ds_train,
        collate_fn=collate_fn,
        batch_sampler=data_util.MultiEpochSampler(batch_sampler, num_epochs),
        num_workers=num_workers,
        pin_memory=True)

    batches_per_epoch = len(batch_sampler)

    return dataloader_train, batches_per_epoch


def _make_dataloader_eval(ds_eval, weights, batch_size, num_workers, distributed_config=None):
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, len(weights), replacement=True)

    if distributed_config is not None:
        sampler = distributed_utils.DistributedSampler(
            sampler, distributed_config.world_size, distributed_config.rank)

    dataloader_eval = torch.utils.data.DataLoader(
        ds_eval,
        collate_fn=functools.partial(
            dataset.collate,
            entity_feature_mapping=ds_eval.node_feature_mapping,
            edge_feature_mapping=ds_eval.edge_feature_mapping),
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True)

    return dataloader_eval


def initialize_datasets(args, distributed_config: distributed_utils.DistributedTrainingInfo = None):
    """Initialize datasets and dataloaders.

    Parameters
    ----------
    args : dict
        Dictionary containing all the dataset configurations.

    distributed_config : distributed_utils.DistributedTrainingInfo, optional
        If not None, configuration options for distributed training.

    Returns
    -------
    torch.data.utils.Dataloader
        Training dataloader
    torch.data.utils.Dataloader
        If not None, testing dataloader
    int
        Number of batches per training epoch
    dataset.EntityFeatureMapping
        Feature mapping in use for entities
    dataset.EdgeFeatureMapping
        Feature mapping in use for constraints
    """
    quantization = {'angle': args['num_quantize_angle'], 'length': args['num_quantize_length']}

    dataset_train_path = args['dataset_train']
    auxiliary_path = args['dataset_auxiliary']

    ds_train, weights_train = load_dataset_and_weights(
        dataset_train_path, auxiliary_path, quantization, args['seed'],
        not args.get('disable_entity_features', False), not args.get('disable_edge_features', False),
        args.get('force_entity_categorical_features', False))

    batch_size = args['batch_size']
    num_workers = args['num_workers']

    if distributed_config:
        batch_size = batch_size // distributed_config.world_size
        num_workers = num_workers // distributed_config.world_size

    collate_fn = functools.partial(
        dataset.collate,
        entity_feature_mapping=ds_train.node_feature_mapping,
        edge_feature_mapping=ds_train.edge_feature_mapping)

    dl_train, batches_per_epoch = make_dataloader_train(
        collate_fn, ds_train, weights_train, batch_size, args['num_epochs'], num_workers, distributed_config)

    if args['dataset_test'] is not None:
        ds_test, weights_test = load_dataset_and_weights_with_mapping(
            args['dataset_test'], ds_train.node_feature_mapping, ds_train.edge_feature_mapping, args['seed'])
        dl_test = _make_dataloader_eval(
            ds_test, weights_test, batch_size, num_workers, distributed_config)
    else:
        dl_test = None

    return dl_train, dl_test, batches_per_epoch, ds_train.node_feature_mapping, ds_train.edge_feature_mapping
