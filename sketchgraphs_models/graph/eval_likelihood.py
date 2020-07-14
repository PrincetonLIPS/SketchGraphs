"""Module to evaluate the graph model according to data likelihood."""

import argparse

import numpy as np
import torch
import tqdm

from sketchgraphs_models import training

from sketchgraphs_models.graph import dataset, sample
from sketchgraphs_models.graph import model as graph_model

from sketchgraphs.data import flat_array


def _total_loss(losses):
    result = 0

    for v in losses.values():
        if v is None:
            continue

        if isinstance(v, dict):
            result += _total_loss(v)
        else:
            result += v.sum()

    return result


def batch_from_example(seq, node_feature_mapping, edge_feature_mapping):
    step_indices = [i for i, op in enumerate(seq) if i > 0 and not dataset._is_subnode_edge(op)]

    batch_list = []

    for step_idx in step_indices:
        graph = dataset.graph_info_from_sequence(seq[:step_idx], node_feature_mapping, edge_feature_mapping)
        target = seq[step_idx]
        batch_list.append((graph, target))

    return dataset.collate(batch_list, node_feature_mapping, edge_feature_mapping)


class GraphLikelihoodEvaluator:
    def __init__(self, model, node_feature_mapping, edge_feature_mapping, device=None):
        self.model = model
        self.node_feature_mapping = node_feature_mapping
        self.edge_feature_mapping = edge_feature_mapping
        self.device = device
        self._feature_dimensions = {
            **node_feature_mapping.feature_dimensions,
            **edge_feature_mapping.feature_dimensions
        }

    def compute_likelihood(self, seqs):
        for i, seq in enumerate(seqs):
            batch = batch_from_example(seq, self.node_feature_mapping, self.edge_feature_mapping)
            batch_device = training.load_cuda_async(batch, device=self.device)

            with torch.no_grad():
                readout = self.model(batch_device)
                losses, *_ = graph_model.compute_losses(readout, batch_device, self._feature_dimensions)
                loss = _total_loss(losses).cpu().item()

            yield loss, len(seq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model_state', help='Path to saved model state_dict.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args()

    device = torch.device(args.device)

    print('Loading trained model')
    model, node_feature_mapping, edge_feature_mapping = sample.load_saved_model(args.model_state)
    model = model.eval().to(device)

    print('Loading testing data')
    seqs = flat_array.load_dictionary_flat(np.load(args.dataset, mmap_mode='r'))['sequences']

    if args.limit is not None:
        seqs = seqs[:args.limit]

    evaluator = GraphLikelihoodEvaluator(model, node_feature_mapping, edge_feature_mapping, device)

    losses = np.empty(len(seqs))
    length = np.empty(len(seqs), dtype=np.int64)

    for i, result in enumerate(tqdm.tqdm(evaluator.compute_likelihood(seqs), total=len(seqs))):
        losses[i], length[i] = result

    print('Average bits per sketch: {:.2f}'.format(losses.mean() / np.log(2)))
    print('Average bits per step: {:.2f}'.format(losses.sum() / np.log(2) / length.sum()))


if __name__ == '__main__':
    main()
