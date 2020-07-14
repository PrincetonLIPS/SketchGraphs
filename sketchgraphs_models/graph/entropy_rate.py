"""Module to compute the entropy rate of a given representation. """

import argparse
import lzma

import numpy as np
import torch
import tqdm

from sketchgraphs_models.graph import dataset, sample
from sketchgraphs.data import flat_array



def sequence_to_integers(seq, node_feature_mapping, edge_feature_mapping):
    graph = dataset.graph_info_from_sequence(seq, node_feature_mapping, edge_feature_mapping)
    all_tensors = (
        [torch.full([1], fill_value=graph.edge_counts[0], dtype=torch.int64)] +
        [graph.node_features,
         graph.edge_features,
         graph.incidence.flatten()] +
        [v.value.flatten() for v in graph.sparse_node_features.values()] +
        [v.value.flatten() for v in graph.sparse_edge_features.values()])

    integers = torch.cat(all_tensors)

    return integers.numpy().astype(np.int32)


def compress_sequences(seqs, node_feature_mapping, edge_feature_mapping):
    compressor = lzma.LZMACompressor(preset=9 | lzma.PRESET_EXTREME)

    total_bytes = 0

    for seq in seqs:
        integers = sequence_to_integers(seq, node_feature_mapping, edge_feature_mapping)
        total_bytes += len(compressor.compress(integers.tobytes()))

    total_bytes += len(compressor.flush())

    return total_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model_state', help='Path to saved model state_dict.')
    parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args()

    print('Loading trained model')
    _, node_feature_mapping, edge_feature_mapping = sample.load_saved_model(args.model_state)

    print('Loading testing data')
    seqs = flat_array.load_dictionary_flat(np.load(args.dataset, mmap_mode='r'))['sequences']

    if args.limit is not None:
        seqs = seqs[:args.limit]

    total_bytes = compress_sequences(tqdm.tqdm(seqs, total=len(seqs)), node_feature_mapping, edge_feature_mapping)
    print('Total bytes: {0}'.format(total_bytes))
    print('Average Entropy: {0:.3f} bits / graph'.format(total_bytes * 8 / len(seqs)))


if __name__ == '__main__':
    main()
