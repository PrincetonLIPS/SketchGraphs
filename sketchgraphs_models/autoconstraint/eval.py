""" Evaluation script for auto-constraint model. """

from __future__ import annotations

import argparse
import dataclasses
import functools
import gzip
import hashlib
import itertools
import os
import pickle
import typing
import queue

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional
from torch import multiprocessing
import tqdm

from sketchgraphs.data import sequence as datalib, flat_array
from sketchgraphs.data import constraint_checks, dof, sketch
from sketchgraphs.pipeline import graph_model as graph_utils
from sketchgraphs.pipeline.graph_model import quantization

from sketchgraphs_models import training
from sketchgraphs_models.autoconstraint import dataset, model as auto_model


class InferenceRequest(typing.NamedTuple):
    node_ops: typing.List[datalib.NodeOp]
    edge_ops: typing.List[datalib.EdgeOp]
    num_nodes_in_graph: int
    partner_index: typing.Optional[int]
    node_features: dict


def _make_valid_constraint_idx_by_types(dof_removed):
    valid_constraint_idx_by_type = {}

    for constraint_label, valid_combinations in dof_removed.items():
        if constraint_label == sketch.ConstraintType.Subnode:
            continue

        constraint_index = dataset.EDGE_IDX_MAP[constraint_label]
        for entity_types in valid_combinations.keys():
            valid_constraint_idx_by_type.setdefault(
                tuple(sorted(entity_types)), []).append(constraint_index)

    return {
        k: torch.tensor(v, dtype=torch.int64).sort().values for k, v in valid_constraint_idx_by_type.items()
    }

_VALID_CONSTRAINT_IDX_BY_TYPE = _make_valid_constraint_idx_by_types(dof.EDGE_DOF_REMOVED)


def mask_from_node_types(node_ops):
    """Creates a mask based on the node types for a given node sequence.

    A mask is a tensor of size `[(num_nodes) * (num_nodes + 1) / 2, num_edge_tyes]`,
    which represents the valid constraints between different nodes in the graph.
    It is formed by concatenating, for each node, tensors representing the valid edge
    types between itself and any of the previous nodes in the graph (including itself),
    where the validity is encoded as having a zero at the given location for valid edge types,
    and a -inf at locations for invalid edge types.
    """
    num_ops = len(node_ops)
    num_constraint_types = len(dataset.EDGE_TYPES_PREDICTED)

    masks = torch.full((num_ops * (num_ops + 1) // 2, num_constraint_types), -np.inf)
    masks[0].fill_(0.0)

    # We start from 1 to skip the external node
    for i in range(1, len(node_ops)):
        mask_offset = i * (i + 1) // 2
        current_label = dof.get_node_label_for_dof(node_ops[i])

        # first node is external, allow all constraints
        masks[mask_offset].fill_(0.0)

        for j in range(1, i + 1):
            key = tuple(sorted((current_label, dof.get_node_label_for_dof(node_ops[j]))))
            masks[mask_offset + j, _VALID_CONSTRAINT_IDX_BY_TYPE[key]] = 0.0

    return masks


def valid_constraint_idx(node_ops, entities, partner, current):
    current_label = dof.get_node_label_for_dof(node_ops[current])
    partner_label = dof.get_node_label_for_dof(node_ops[partner])

    valid_idx_by_type = _VALID_CONSTRAINT_IDX_BY_TYPE[tuple(sorted((current_label, partner_label)))]

    valid_idx = []

    for idx in valid_idx_by_type:
        edge_label = dataset.EDGE_TYPES_PREDICTED[idx]
        constraint_fn = constraint_checks.CONSTRAINT_BY_LABEL.get(edge_label)

        if constraint_fn is None:
            # Unknown check, assume valid.
            valid_idx.append(idx)
            continue

        if partner == current:
            check = constraint_fn(entities[current])
        else:
            check = constraint_fn(entities[current], entities[partner])

        if check or check is None:
            valid_idx.append(idx)

    return torch.tensor(valid_idx, dtype=torch.int64)


def mask_from_satisfied(node_ops):
    num_ops = len(node_ops)
    num_constraint_types = len(dataset.EDGE_TYPES_PREDICTED)

    masks = torch.full((num_ops * (num_ops + 1) // 2, num_constraint_types), -np.inf)
    masks[0].fill_(0.0)

    entities = [None] + [constraint_checks.get_entity_by_idx(node_ops, i) for i in range(1, len(node_ops))]

    for i in range(1, len(node_ops)):
        mask_offset = i * (i + 1) // 2

        # first node is external, allow all constraints
        masks[mask_offset].fill_(0.0)

        for j in range(1, i + 1):
            masks[mask_offset + j, valid_constraint_idx(node_ops, entities, j, i)] = 0.0

    return masks


class ConstraintGenerator:
    node_ops: list[datalib.NodeOp]
    edge_ops: list[datalib.EdgeOp]

    def __init__(self, node_ops, max_edges=100, seed=2, node_features=None, mask=None):
        assert len(node_ops) >= 2
        self.node_ops = node_ops
        self.edge_ops = []
        self.current_op = 1
        self.partner_index = None
        self.max_edges = max_edges
        self.seed = seed
        self._rng = None
        self.node_features = node_features
        self.mask = mask

    @property
    def rng(self):
        if self._rng is None:
            self._rng = torch.Generator().manual_seed(
                int.from_bytes(hashlib.sha256(self.seed.to_bytes(4, 'little')).digest()[:4], 'little'))
        return self._rng

    @property
    def current_request(self):
        return InferenceRequest(
            self.node_ops, self.edge_ops,
            self.current_op + 1, self.partner_index,
            self.node_features)

    @property
    def current_nodes_in_graph(self):
        return self.current_op + 1

    def _fixup_subnode(self):
        if len(self.node_ops) == self.current_op:
            return

        current_node = self.node_ops[self.current_op]
        if not isinstance(current_node.label, datalib.SubnodeType):
            # Not currently looking at subnode, just skip
            return

        for i in reversed(range(self.current_op)):
            if isinstance(self.node_ops[i].label, datalib.SubnodeType):
                continue

            self.edge_ops.append(datalib.EdgeOp(datalib.ConstraintType.Subnode, (self.current_op, i)))
            break
        else:
            raise RuntimeError('Could not find node corresponding to subnode')


    def _stop_current_node(self):
        self.current_op += 1
        if self.current_op == len(self.node_ops):
            # Completely done with all edges, signal to stop.
            return True
        else:
            # Done with current node, move to next one and fix-up subnode if necessary.
            self._fixup_subnode()
            return False


    def next_label(self, label_logits):
        if self.partner_index is None:
            raise ValueError('Invalid construction step, infer partner before inferring label.')

        assert label_logits.shape[0] == len(dataset.EDGE_TYPES_PREDICTED)

        probs = torch.nn.functional.softmax(label_logits, dim=0)
        label = int(torch.multinomial(probs, 1, generator=self.rng))
        self.edge_ops.append(datalib.EdgeOp(
            dataset.EDGE_TYPES_PREDICTED[label],
            (self.current_op, self.partner_index)))
        self.partner_index = None

        return len(self.edge_ops) == self.max_edges

    def next_partner(self, partner_logits):
        if self.partner_index is not None:
            raise ValueError('Invalid construction step, label inference pending.')

        assert partner_logits.shape[0] == self.current_op + 1

        stop_prob = torch.exp(-torch.nn.functional.softplus(torch.logsumexp(partner_logits, 0)))

        should_stop = torch.bernoulli(stop_prob, generator=self.rng)

        if should_stop:
            return self._stop_current_node()

        probs = torch.nn.functional.softmax(partner_logits, dim=0)
        partner_idx = int(torch.multinomial(probs, 1, generator=self.rng))
        self.partner_index = partner_idx
        return False

    def next_joint(self, partner_logits, conditional_label_logits):
        assert tuple(conditional_label_logits.shape) == (self.current_op + 1, len(dataset.EDGE_TYPES_PREDICTED))
        if self.partner_index is not None:
            raise ValueError('Invalid construction step, label inference pending.')

        partner_logits_log_norm = torch.nn.functional.softplus(torch.logsumexp(partner_logits, 0))

        partner_logits = partner_logits - partner_logits_log_norm
        conditional_label_logits = torch.nn.functional.log_softmax(conditional_label_logits, dim=-1)

        joint_prob = partner_logits.unsqueeze(-1) + conditional_label_logits

        if self.mask is not None:
            num_nodes = self.current_nodes_in_graph
            joint_prob += torch.narrow(self.mask, 0, num_nodes * (num_nodes - 1) // 2, num_nodes)

        log_remaining_prob = torch.logsumexp(joint_prob, (0, 1))
        log_renorm = (
            torch.nn.functional.softplus(-torch.abs(partner_logits_log_norm + log_remaining_prob)) +
            torch.max(-partner_logits_log_norm, log_remaining_prob))

        stop_prob = torch.exp(-partner_logits_log_norm - log_renorm)

        should_stop = torch.bernoulli(stop_prob, generator=self.rng)
        if should_stop:
            return self._stop_current_node()

        probs = torch.exp(joint_prob.contiguous()).flatten()
        sample_idx = int(torch.multinomial(probs, 1, generator=self.rng))

        partner_idx, label_idx = divmod(sample_idx, len(dataset.EDGE_TYPES_PREDICTED))
        self.edge_ops.append(
            datalib.EdgeOp(dataset.EDGE_TYPES_PREDICTED[label_idx], (self.current_op, partner_idx)))


@dataclasses.dataclass(order=True)
class PriorityItem:
    priority: int
    item: typing.Any = dataclasses.field(compare=False)



def _node_features(node_ops: Sequence[datalib.NodeOp], node_feature_mappings: quantization.EntityFeatureMapping):
    all_node_labels = torch.tensor([dataset.NODE_IDX_MAP[op.label] for op in node_ops], dtype=torch.int64)
    sparse_node_features = node_feature_mappings.all_sparse_features(node_ops)

    return {
        'node_features': all_node_labels.share_memory_(),
        'sparse_node_features': dataset._sparse_feature_to_torch(sparse_node_features)
    }


def _make_generator_from_nodes(args: tuple[int, Sequence[datalib.NodeOp]],
                               max_edges: int, seed: int, node_feature_mappings: quantization.EntityFeatureMapping,
                               mask_function) -> tuple[int, Optional[ConstraintGenerator]]:
    seq_idx, node_ops = args

    if node_ops[-1].label == datalib.EntityType.Stop:
        node_ops = node_ops[:-1]
    if len(node_ops) < 2:
        return seq_idx, None

    cg = ConstraintGenerator(
        node_ops,
        max_edges=max_edges,
        seed=seed + seq_idx,
        node_features=_node_features(node_ops, node_feature_mappings),
        mask=mask_function(node_ops).share_memory_() if mask_function is not None else None)

    return seq_idx, cg


class AutoConstraintPrediction:
    def __init__(self, model, node_feature_mappings, batch_size=32, seed=2, device=None, max_edges=100, mask_function=None):
        self.model = model
        self.node_feature_mappings = node_feature_mappings
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.max_edges = max_edges
        self.mask_function = mask_function

    def _node_features(self, node_ops):
        all_node_labels = torch.tensor([dataset.NODE_IDX_MAP[op.label] for op in node_ops], dtype=torch.int64)
        sparse_node_features = self.node_feature_mappings.all_sparse_features(node_ops)
        return {
            'node_features': all_node_labels,
            'sparse_node_features': sparse_node_features
        }


    def _readout_batch(self, current_batch: Sequence[tuple[int, ConstraintGenerator]], compute_all_label_logits=False):
        requests = [g.current_request for _, g in current_batch]
        request_features = [feature_from_request(r) for r in requests]
        batch = dataset.collate(request_features)

        batch_device = training.load_cuda_async(batch, device=self.device)

        with torch.no_grad():
            readout = self.model(batch_device, compute_all_label_logits)

        edge_partner_logits: torch.Tensor = readout['edge_partner_logits'].detach().cpu()
        edge_label_logits: torch.Tensor = readout['edge_label_logits'].detach().cpu()

        return batch, edge_label_logits, edge_partner_logits


    def _process_batch_factorized(self, current_batch: Sequence[tuple[int, ConstraintGenerator]]) -> list[int]:
        batch, edge_label_logits, edge_partner_logits = self._readout_batch(current_batch)

        graph = batch['graph']

        sorted_indices = batch['sorted_indices']
        reverse_indices = torch.argsort(sorted_indices)
        partner_index_index = batch['partner_index'].index

        generators_to_finalize: list[int] = []
        cg_index_inspected = []

        for i, (batch_idx, (_, cg)) in enumerate(zip(reverse_indices, current_batch)):
            if cg.partner_index is not None:
                # Skip label requests for now
                continue

            logits = torch.narrow(
                edge_partner_logits, 0, graph.node_offsets[batch_idx], graph.node_counts[batch_idx])
            if cg.next_partner(logits):
                generators_to_finalize.append(i)
            cg_index_inspected.append(i)

        for i in range(edge_label_logits.shape[0]):
            logits = edge_label_logits[i]
            batch_idx = partner_index_index[i]
            cg_idx = sorted_indices[batch_idx]
            _, cg = current_batch[cg_idx]

            if cg.next_label(logits):
                generators_to_finalize.append(cg_idx)
            cg_index_inspected.append(cg_idx)

        return generators_to_finalize


    def _process_batch_joint(self, current_batch: Sequence[tuple[int, ConstraintGenerator]]) -> list[int]:
        batch, edge_label_logits, edge_partner_logits = self._readout_batch(
            current_batch, compute_all_label_logits=True)

        graph = batch['graph']
        sorted_indices = batch['sorted_indices']
        reverse_indices = torch.argsort(sorted_indices)

        generators_to_finalize: list[int] = []

        for i, (_, cg) in enumerate(current_batch):
            batch_idx = int(reverse_indices[i])
            partner_logits = torch.narrow(
                edge_partner_logits, 0, graph.node_offsets[batch_idx], graph.node_counts[batch_idx])
            label_logits = torch.narrow(
                edge_label_logits, 0, graph.node_offsets[batch_idx], graph.node_counts[batch_idx])
            if cg.next_joint(partner_logits, label_logits):
                generators_to_finalize.append(i)

        return generators_to_finalize

    def predict(self, node_seqs, use_joint=False, num_workers=0):
        _make_generator = functools.partial(
            _make_generator_from_nodes,
            max_edges=self.max_edges,
            seed=self.seed,
            node_feature_mappings=self.node_feature_mappings,
            mask_function=self.mask_function)

        if num_workers != 0:
            pool = multiprocessing.Pool(num_workers)
            cg_seqs = pool.imap(_make_generator, enumerate(node_seqs), chunksize=4)
        else:
            cg_seqs = map(_make_generator, enumerate(node_seqs))

        iter_cg_seqs = iter(cg_seqs)
        node_seqs_done = False
        current_batch: list[tuple[int, ConstraintGenerator]] = []
        output_buffer = queue.PriorityQueue()
        next_output_index = 0

        while len(current_batch) > 0 or not node_seqs_done:
            while len(current_batch) < self.batch_size and not node_seqs_done:
                try:
                    seq_idx, cg = next(iter_cg_seqs)
                except StopIteration:
                    node_seqs_done = True
                    break

                if cg is None:
                    output_buffer.put(PriorityItem(seq_idx, []))
                    continue

                current_batch.append((seq_idx, cg))

            if use_joint:
                generators_to_finalize = self._process_batch_joint(current_batch)
            else:
                generators_to_finalize = self._process_batch_factorized(current_batch)
            generators_to_finalize.sort()

            for i in reversed(generators_to_finalize):
                seq_idx, cg = current_batch.pop(i)
                result = cg.edge_ops

                if seq_idx == next_output_index:
                    next_output_index += 1
                    yield result
                else:
                    output_buffer.put(PriorityItem(seq_idx, result))
                    while not output_buffer.empty() and output_buffer.queue[0].priority == next_output_index:
                        next_output_index += 1
                        yield output_buffer.get().item

        while not output_buffer.empty():
            queued = output_buffer.get()
            assert queued.priority == next_output_index
            next_output_index += 1
            yield queued.item



def feature_from_request(request: InferenceRequest):
    result = {**request.node_features}

    edge_labels = torch.tensor([dataset.EDGE_IDX_MAP[op.label] for op in request.edge_ops], dtype=torch.int64)

    if len(request.edge_ops) > 0:
        incidence = torch.tensor([(op.references[0], op.references[-1]) for op in request.edge_ops],
                                 dtype=torch.int64).T.contiguous()
        incidence = torch.cat((incidence, torch.flip(incidence, [0])), dim=1)
    else:
        incidence = torch.empty([2, 0], dtype=torch.int64)

    edge_features = edge_labels.repeat(2)

    assert request.num_nodes_in_graph <= len(result['node_features'])
    result['graph'] = graph_utils.GraphInfo.from_single_graph(incidence, None, edge_features, request.num_nodes_in_graph)

    result['partner_index'] = request.partner_index if request.partner_index is not None else -1

    # Unused, present for compatibility
    result['target_edge_label'] = -1 if request.partner_index is None else 0

    return result


def load_sampling_model(model_path):
    state_dict = torch.load(model_path, map_location='cpu')

    node_feature_mapping = quantization.EntityFeatureMapping()
    node_feature_mapping.load_state_dict(state_dict['node_feature_mapping'])

    model_config = state_dict['model_configuration']

    model = auto_model.AutoconstraintModel(
        auto_model.MODEL_CORES[model_config.get('model_core', 'global_embedding')](
            model_config['embedding_dim'],
            node_feature_mapping.feature_dimensions,
            model_config['depth']))
    model.load_state_dict(state_dict['model'])

    return model, node_feature_mapping


def split_ops(ops):
    node_ops: list[datalib.NodeOp] = []
    edge_ops: list[datalib.EdgeOp] = []

    for op in ops:
        if isinstance(op, datalib.EdgeOp):
            edge_ops.append(op)
        else:
            node_ops.append(op)

    return node_ops, edge_ops


MASK_FUNCTIONS = {
    'node_type': mask_from_node_types,
    'satisfied': mask_from_satisfied,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output', type=str)
    parser.add_argument('--output_statistics', type=str)
    parser.add_argument('--max_edge', type=int, default=100)
    parser.add_argument('--max_predictions', type=int, default=None)
    parser.add_argument('--use_joint', action='store_true')
    parser.add_argument('--mask', default=None, choices=list(MASK_FUNCTIONS.keys()))
    parser.add_argument('--num_workers', default=0, type=int)

    args = parser.parse_args()

    device = torch.device(args.device)

    print('Loading trained model')
    model, node_feature_mapping = load_sampling_model(args.model)
    model = model.eval().to(device)

    print('Loading testing data')
    seqs = flat_array.load_dictionary_flat(np.load(args.dataset, mmap_mode='r'))['sequences']

    prediction = AutoConstraintPrediction(
        model, node_feature_mapping,
        batch_size=args.batch_size, device=device,
        mask_function=MASK_FUNCTIONS[args.mask] if args.mask is not None else None)

    length = len(seqs)
    if args.max_predictions is not None:
        length = min(length, args.max_predictions)

    input_seq_prediction, input_seq_verification = itertools.tee(
        (seqs[i] for i in range(length)), 2)

    input_node_ops = ([op for op in seq if isinstance(op, datalib.NodeOp)] for seq in input_seq_prediction)
    prediction_output = prediction.predict(input_node_ops, use_joint=args.use_joint, num_workers=args.num_workers)


    precision = np.empty(length, dtype=np.float64)
    recall = np.empty(length, dtype=np.float64)
    ops = []

    for i, (predicted_edge_ops, original_ops) in enumerate(tqdm.tqdm(zip(prediction_output, input_seq_verification), total=length)):
        node_ops, edge_ops = split_ops(original_ops)
        ops.append({
            'node_ops': node_ops,
            'edge_ops': edge_ops,
            'predicted_edge_ops': predicted_edge_ops,
        })

        predicted_edge_ops = set((e.label, e.references[0], e.references[-1])
                                  for e in predicted_edge_ops if e.label != sketch.ConstraintType.Subnode)
        edge_ops = set((e.label, e.references[0], e.references[-1])
                        for e in edge_ops if e.label != sketch.ConstraintType.Subnode)

        num_correct_edge_ops = len(edge_ops & predicted_edge_ops)
        precision[i] = num_correct_edge_ops / len(predicted_edge_ops) if len(predicted_edge_ops) > 0 else 0
        recall[i] = num_correct_edge_ops / len(edge_ops) if len(edge_ops) > 0 else 1

    print('Average recall {}. Average precision {}'.format(np.mean(recall), np.mean(precision)))

    if args.output is not None:
        with gzip.open(args.output, 'wb') as f:
            pickle.dump(ops, f, protocol=4)

    output_statistics_file = args.output_statistics

    if output_statistics_file is None:
        if args.output is not None:
            output_basename, output_ext = os.path.splitext(args.output)
            if output_ext == '.gz':
                output_basename, _ = os.path.splitext(output_basename)

            output_statistics_file = output_basename + '_stat.npz'

    if output_statistics_file is not None:
        np.savez_compressed(output_statistics_file, precision=precision, recall=recall)



def load_pickle_ops(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
