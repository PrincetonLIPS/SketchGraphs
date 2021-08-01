"""Evaluates an auto-constraint model in terms of the sequential assigned likelihood. """

import argparse
import itertools
import gzip
import pickle

import numpy as np
import torch
import tqdm

from sketchgraphs import data as datalib
from sketchgraphs.data import flat_array

from sketchgraphs_models import training
from sketchgraphs_models.autoconstraint import dataset, model as auto_model, eval


def _edge_ops(ops):
    return [op for op in ops if isinstance(op, datalib.EdgeOp)]

def _node_ops(ops):
    return [op for op in ops if isinstance(op, datalib.NodeOp)]


def ops_to_batch(ops, node_feature_mappings):
    """Given a sequence of operations representing a sketch, constructs a batch
    """
    node_ops = [op for op in ops if isinstance(op, datalib.NodeOp)]

    batch = []

    for i, op in enumerate(ops):
        if i == 0:
            # First external node does not induce edge stop token
            continue

        if op.label == datalib.ConstraintType.Subnode:
            continue

        ops_in_graph = ops[:i]

        features = dataset.process_node_and_edge_ops(
            node_ops, _edge_ops(ops_in_graph),
            len(_node_ops(ops_in_graph)), node_feature_mappings)

        if isinstance(op, datalib.NodeOp):
            # Stop problem
            partner_index = -1
            target_edge_label = -1
        else:
            partner_index = op.references[-1]
            target_edge_label = dataset.EDGE_IDX_MAP[op.label]

        features['partner_index'] = partner_index
        features['target_edge_label'] = target_edge_label

        batch.append(features)

    # Append final edge stop target
    batch.append({
        **dataset.process_node_and_edge_ops(node_ops, _edge_ops(ops), len(node_ops), node_feature_mappings),
        'partner_index': -1,
        'target_edge_label': -1
        })

    return batch


class EdgeLikelihoodEvaluator:
    def __init__(self, model, node_feature_mappings, device=None):
        self.model = model
        self.node_feature_mappings = node_feature_mappings
        self.device = device

    def edge_likelihood(self, ops):
        if ops[-1].label == 'Stop':
            ops = ops[:-1]

        if len(ops) == 1:
            return np.empty([0, 2])

        batch_list = ops_to_batch(ops, self.node_feature_mappings)

        batch = dataset.collate(batch_list)
        batch_device = training.load_cuda_async(batch, self.device)

        with torch.no_grad():
            readout = self.model(batch_device)
            losses, _ = auto_model.compute_losses(batch_device, readout, reduction='none')

        losses_flat = torch.zeros((len(batch_list), 2))
        losses_flat[batch['partner_index'].index, 0] = losses['edge_partner'].cpu()
        losses_flat[batch['stop_partner_index_index'], 0] = losses['edge_stop'].cpu()

        losses_flat[batch['partner_index'].index, 1] = losses['edge_label'].cpu()

        losses_flat = losses_flat[torch.argsort(batch['sorted_indices'])]

        return losses_flat.numpy()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output', type=str)
    parser.add_argument('--max_predictions', type=int, default=None)


    args = parser.parse_args()

    device = torch.device(args.device)

    print('Loading trained model')
    model, node_feature_mapping = eval.load_sampling_model(args.model)
    model = model.eval().to(device)

    print('Loading testing data')
    seqs = flat_array.load_dictionary_flat(np.load(args.dataset, mmap_mode='r'))['sequences']

    likelihood_evaluation = EdgeLikelihoodEvaluator(model, node_feature_mapping, device)

    length = len(seqs)
    if args.max_predictions is not None:
        length = min(length, args.max_predictions)

    results = []
    average_likelihoods = np.empty(length)
    sequence_length = np.empty(length)

    for i in tqdm.trange(length):
        seq = seqs[i]
        result = likelihood_evaluation.edge_likelihood(seq)
        results.append({
            'seq': seq,
            'likelihood': result
        })

        sequence_length[i] = result.shape[0]

        if result.shape[0] == 0:
            average_likelihoods[i] = 0.0
        else:
            average_likelihoods[i] = np.mean(np.sum(result, axis=-1), axis=0)

    print('Average bit per edge {0:.3f}'.format(np.average(average_likelihoods, weights=sequence_length) / np.log(2)))

    if args.output is None:
        return

    print('Saving to output {0}'.format(args.output))
    with gzip.open(args.output, 'wb') as f:
        pickle.dump(results, f, protocol=4)


if __name__ == '__main__':
    main()
