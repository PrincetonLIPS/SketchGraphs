"""This module implements sampling from the entity model.
"""

import argparse
import gzip
import json
import enum
import multiprocessing as mp
import os
import pickle

import torch
import numpy as np
import tqdm

from sketchgraphs.data import sketch as datalib
from sketchgraphs.data.sequence import NodeOp, EdgeOp
from sketchgraphs.data.sketch import EntityType
from sketchgraphs.pipeline import graph_model as graph_utils
from sketchgraphs.pipeline.graph_model import target

from sketchgraphs_models import training
from sketchgraphs_models.graph import dataset, model as graph_model
from sketchgraphs_models.nn import functional as sg_functional

from .train import make_model_with_arguments

# pylint: disable=no-member


def _onehot(index, depth):
    out = torch.zeros(depth)
    out[index] = 1
    return out


class _SeqBuilder(object):
    def __init__(self):
        self.seq = [NodeOp(datalib.EntityType.External)]
        self.last_entity_node = None
        self.num_nodes = 1

    def add_op(self, op):
        if isinstance(op, NodeOp):
            return self._add_node(op.label, op.parameters)
        if isinstance(op, EdgeOp):
            self.seq.append(op)

    def _add_node(self, label, parameters=None):
        if parameters is None:
            parameters = {}

        self.seq.append(NodeOp(label, parameters))
        self.num_nodes += 1

        if not isinstance(label, datalib.SubnodeType):
            self.last_entity_node = self.num_nodes - 1
            return

        # Add subedge
        self.add_op(EdgeOp(datalib.ConstraintType.Subnode, (self.num_nodes - 1, self.last_entity_node)))


class GraphSamplingModel(torch.nn.Module):
    def __init__(self, model_core, entity_label, entity_feature_readout, edge_post_embedding, edge_label,
                 edge_feature_readout, edge_partner, feature_dimensions):
        super(GraphSamplingModel, self).__init__()
        self.model_core = model_core
        self.entity_label = entity_label
        self.entity_feature_readout = entity_feature_readout
        self.edge_post_embedding = edge_post_embedding
        self.edge_label = edge_label
        self.edge_partner = edge_partner
        self.edge_feature_readout = edge_feature_readout
        self.feature_dimensions = feature_dimensions

    @staticmethod
    def from_numerical_model(model: graph_model.GraphModel, feature_dimensions):
        """Creates a new sampling model from the given numerical model. """
        return GraphSamplingModel(
            model.model_core,
            model.entity_label,
            model.entity_feature_readout,
            model.edge_post_embedding,
            model.edge_label,
            model.edge_feature_readout,
            model.edge_partner,
            feature_dimensions)

    def sample_node_label(self, graph, generator=None):
        """Samples node labels.

        This function samples a node label for the last node in each graph.

        Parameters
        ----------
        graph : GraphInfo
            The graph (or batch of graph for which to obtain node labels).
        generator : torch.Generator, optional
            Optional PRNG to use for sampling.

        Returns
        -------
        torch.Tensor
            An integer tensor containing a sampled node label for each graph in the batch.
        """
        data = {'graph': graph}
        _, graph_embedding = self.model_core(data)
        node_label_logits = self.entity_label(graph_embedding)

        node_label_samples = torch.multinomial(
            torch.nn.functional.softmax(node_label_logits, dim=1),
            1, replacement=True, generator=generator).squeeze(-1)

        return node_label_samples


    def sample_entity_features(self, graph, target_type, features, generator=None):
        """Samples entity features.

        Parameters
        ----------
        graph : GraphInfo
            The graph (or batch of graphs) for which to obtain node features
        target_type : TargetType
            The target type / label of the entity for which to sample features
        features : torch.Tensor
            An integer tensor corresponding to the entity features
        generator : torch.Generator, optional
            Optional PRNG to use for sampling

        Returns
        -------
        torch.Tensor
            An integer tensor containing the sampled node features
        """
        _, graph_embedding = self.model_core({'graph': graph})

        readout = self.entity_feature_readout[target_type.name]
        node_feature_logits = readout(features, graph_embedding)
        feature_dimension = self.feature_dimensions[target_type]

        labels = []

        current_offset = 0
        for dim in feature_dimension.values():
            labels.append(
                torch.multinomial(
                    torch.nn.functional.softmax(
                        torch.narrow(node_feature_logits, 1, current_offset, dim), dim=1),
                    1, replacement=True, generator=generator).squeeze(-1))
            current_offset += dim

        return torch.cat(labels, dim=-1)


    def sample_edge_target(self, graph, generator=None):
        """Samples edge targets.

        This function samples potential edges between the last node and a given target.
        The sampled target may potentially be one past the end of the nodes, indicating
        that no new edges should be created.

        Parameters
        ----------
        graph : GraphInfo
            the graph (or batch of graphs) for which to obtain node labels.
        generator : torch.Generator, optional
            Optional PRNG to use for sampling

        Returns
        -------
        torch.Tensor
            An integer tensor containing the sampled edge target for each graph in the batch.
        """
        data = {'graph': graph}
        node_embedding, graph_embedding = self.model_core(data)
        partner_logits = self.edge_partner(node_embedding, graph_embedding, graph)

        partner_target_samples = sg_functional.segmented_multinomial_extended(
            partner_logits, graph_utils.scopes_from_offsets(graph.node_offsets),
            generator=generator)

        return partner_target_samples

    def sample_edge_label(self, graph, targets, generator=None):
        """Samples edge labels.

        This function samples edge label types for given edges between the last node in the graph
        and the specified target node.

        Parameters
        ----------
        graph : GraphInfo
            the graph (or batch of graphs) for which to obtain edge labels
        targets : torch.Tensor
            A tensor indicating the indices of the edges
        generator : torch.Generator, optional
            Optional PRNG to use for sampling

        Returns
        -------
        torch.Tensor
            An integer tensor containing the sampled edge labels.
        """
        node_post_embedding, graph_embedding = self.model_core({'graph': graph})
        current_node_post_embedding = node_post_embedding.index_select(0, graph.node_offsets[1:] - 1)
        partner_node_post_embedding = node_post_embedding.index_select(0, targets)

        edge_post_embedding = self.edge_post_embedding(current_node_post_embedding, partner_node_post_embedding)

        edge_label_logits = self.edge_label(edge_post_embedding, graph_embedding)

        edge_labels = torch.multinomial(
            torch.nn.functional.softmax(edge_label_logits, dim=1),
            1, replacement=True, generator=generator).squeeze(-1)

        return edge_labels


    def sample_edge_features(self, graph, targets, target_type, numerical_features, generator=None):
        node_post_embedding, graph_embedding = self.model_core({'graph': graph})
        current_node_post_embedding = node_post_embedding.index_select(0, graph.node_offsets[1:] - 1)
        partner_node_post_embedding = node_post_embedding.index_select(0, targets)

        edge_post_embedding = self.edge_post_embedding(
            current_node_post_embedding, partner_node_post_embedding)
        edge_feature_logits = self.edge_feature_readout[target_type.name](
            numerical_features, edge_post_embedding, graph_embedding)
        feature_dimension = self.feature_dimensions[target_type]

        labels = []

        current_offset = 0
        for dim in feature_dimension.values():
            labels.append(
                torch.multinomial(
                    torch.nn.functional.softmax(
                        torch.narrow(edge_feature_logits, 1, current_offset, dim), dim=1),
                    1, replacement=True, generator=generator).squeeze(-1))
            current_offset += dim

        return torch.cat(labels, dim=-1)


def _get_subnodes_for_entity(node_op):
    label = node_op.label
    if label in (datalib.EntityType.External, datalib.EntityType.Conic):
        return []
    entity_cls = datalib.ENTITY_TYPE_TO_CLASS[label]

    return entity_cls.get_subnode_types()


def _sample_nodes(model: GraphSamplingModel, graph, node_feature_mapping, generator=None):
    labels_idx = model.sample_node_label(graph, generator)

    if len(labels_idx) != 1:
        raise NotImplementedError('Batch sampling is not implemented')

    label_idx = int(labels_idx[0])
    label = dataset.NODE_IDX_MAP_REVERSE[label_idx]

    target_type = target.TargetType.from_label(label)

    if target_type not in model.feature_dimensions:
        return [NodeOp(label)]

    if target_type in model.feature_dimensions:
        fd = model.feature_dimensions[target_type]
        numerical_features = labels_idx.new_zeros((1, len(fd)), dtype=int)

        for feat_idx in range(len(fd)):
            sample_features = model.sample_entity_features(
                graph, target_type, numerical_features, generator)
            numerical_features[0, feat_idx] = sample_features[feat_idx]

        numerical_features = numerical_features[0].cpu().tolist()
        parameters = node_feature_mapping.features_from_index(numerical_features, target_type)
    else:
        parameters = {}

    return [NodeOp(label, parameters)]


def _sample_edges(model: GraphSamplingModel, graph, edge_feature_mapping, generator=None):
    partners = model.sample_edge_target(graph, generator)

    if len(partners) != 1:
        raise NotImplementedError('Batch sampling is not implemented')

    idx_targets = np.flatnonzero((partners < graph.node_counts).cpu().numpy())

    result = [None] * partners.shape[0]

    if len(idx_targets) == 0:
        return result

    graph_with_targets = graph
    partners += graph_with_targets.node_offsets[:-1]
    label_indices = model.sample_edge_label(graph_with_targets, partners, generator)

    # Edge feature sampling
    target_type = target.TargetType.from_label(target.EDGE_IDX_MAP_REVERSE[int(label_indices[0])])
    numerical_features = None
    if target_type in model.feature_dimensions:
        fd = model.feature_dimensions[target_type]
        numerical_features = label_indices.new_zeros((1, len(fd)), dtype=int)
        for feat_idx in range(len(fd)):
            sampled_features = model.sample_edge_features(
                graph_with_targets, partners, target_type, numerical_features, generator)
            numerical_features[0, feat_idx] = sampled_features[feat_idx]


    node_counts = graph_with_targets.node_counts.tolist()

    for i, idx in enumerate(idx_targets):
        label_idx = int(label_indices[i])
        label = target.EDGE_IDX_MAP_REVERSE[label_idx]
        target_type = target.TargetType.from_label(label)

        if numerical_features is not None:
            parameters_indices = numerical_features[i].cpu().tolist()
            parameters = edge_feature_mapping.features_from_index(parameters_indices, target_type)
            for param_id, val in parameters.items():
                if isinstance(val, enum.IntEnum):
                    parameters[param_id] = val.name
        else:
            parameters = {}

        partner_idx = int(partners[idx])

        if partner_idx == int(node_counts[i]) - 1:
            references = (partner_idx,)
        else:
            references = int(node_counts[i] - 1), partner_idx

        result[idx] = EdgeOp(label, references, parameters)

    return result


def generate_sample(model: GraphSamplingModel, max_iters, node_feature_mapping, edge_feature_mapping, generator=None, device=None):
    builder = _SeqBuilder()
    state = 'add_node'
    subnodes_to_add = None

    while len(builder.seq) < max_iters:
        graph = dataset.graph_info_from_sequence(builder.seq, node_feature_mapping, edge_feature_mapping)
        graph = training.load_cuda_async(graph, device)

        if state == 'add_node':
            node_op, = _sample_nodes(model, graph, node_feature_mapping, generator)
            builder.add_op(node_op)
            if node_op.label == EntityType.Stop:
                break

            subnodes_to_add = list(_get_subnodes_for_entity(node_op))
            state = 'add_edge'
        elif state == 'add_edge':
            edge_op, = _sample_edges(model, graph, edge_feature_mapping, generator)
            if edge_op is not None:
                assert max(edge_op.references) + 1 == builder.num_nodes
                builder.add_op(edge_op)
                continue

            if subnodes_to_add:
                subnode_op = NodeOp(subnodes_to_add.pop())
                builder.add_op(subnode_op)
                state = 'add_edge'
            else:
                state = 'add_node'
        else:
            assert False
    return builder.seq


def load_saved_model(model_state_path):
    state = torch.load(model_state_path, map_location='cpu')

    args_path = os.path.join(os.path.dirname(model_state_path), 'args.txt')
    with open(args_path) as fh:
        model_args = json.load(fh)

    if state['node_feature_mapping'] is not None:
        node_feature_mapping = dataset.EntityFeatureMapping()
        node_feature_mapping.load_state_dict(state['node_feature_mapping'])
    else:
        node_feature_mapping = None

    if state['edge_feature_mapping'] is not None:
        edge_feature_mapping = dataset.EdgeFeatureMapping()
        edge_feature_mapping.load_state_dict(state['edge_feature_mapping'])
    else:
        edge_feature_mapping = None

    feature_dimensions = {**node_feature_mapping.feature_dimensions, **edge_feature_mapping.feature_dimensions}
    model = make_model_with_arguments(feature_dimensions, model_args)

    ## Remove "module." from beginning of keys
    new_state_dict = {}
    for key in state['model']:
        if key.startswith('module.'):
            new_state_dict[key[7:]] = state['model'][key]
        else:
            new_state_dict[key] = state['model'][key]

    model.load_state_dict(new_state_dict)

    return model, node_feature_mapping, edge_feature_mapping


def _load_sampling_model(args):
    model, node_feature_mapping, edge_feature_mapping = load_saved_model(args.model_state)
    feature_dimensions = {**node_feature_mapping.feature_dimensions, **edge_feature_mapping.feature_dimensions}
    model = model.to(args.device)

    return GraphSamplingModel.from_numerical_model(model, feature_dimensions), node_feature_mapping, edge_feature_mapping

def sample_iterator(args):
    _set_rng_seeds(args.seed)
    sampling_model, node_feature_mapping, edge_feature_mapping = _load_sampling_model(args)
    for _ in range(args.num_samples):
        yield generate_sample(sampling_model, args.max_iters, node_feature_mapping, edge_feature_mapping, device=torch.device(args.device))

def sample_and_print(args):
    for i, seq in enumerate(sample_iterator(args)):
        print('sample #%d' % i)
        for j, op in enumerate(seq):
            print(j, op)
        print('\n')

def sample_and_write_pickle(args, output_path, proc_idx):
    args.seed = args.seed + proc_idx
    seqs = sample_iterator(args)
    seqs = tqdm.tqdm(sample_iterator(args), total=args.num_samples)
    seqs = list(seqs)

    with gzip.open(output_path, 'wb') as fh:
        pickle.dump(seqs, fh)


def _sample_pickle_with_counter(args, process_id, result_queue):
    args.seed = args.seed + process_id
    args.num_samples = args.num_samples // args.num_procs
    print('Worker {0} loading data model and sampling {1} samples'.format(process_id, args.num_samples))

    seqs = []
    for i, seq in enumerate(sample_iterator(args)):
        seqs.append(seq)

        if i % 10 == 0:
            result_queue.put(seqs)
        seqs = []

    result_queue.put(seqs)
    result_queue.put(None)


def sample_and_write_pickle_multithreaded(args):
    print('Sampling with {0} workers'.format(args.num_procs))
    result_queue = mp.Queue()
    workers = [mp.Process(target=_sample_pickle_with_counter, args=(args, i, result_queue)) for i in range(args.num_procs)]

    for worker in workers:
        worker.start()

    num_workers_remaining = len(workers)
    seqs = []

    with tqdm.tqdm(total=args.num_samples) as progress:
        while num_workers_remaining > 0:
            result = result_queue.get()

            if result is None:
                num_workers_remaining -= 1
                continue

            seqs.extend(result)
            progress.update(len(result))

    with gzip.open(args.output_path, 'wb') as out_file:
        pickle.dump(seqs, out_file)


def _set_rng_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', help='Path for output pickle file.')
    parser.add_argument('--model_state', help='Path to saved model state_dict.')
    parser.add_argument('--model_args_file', help='Path to saved model args file. If not provided will be inferred.')
    parser.add_argument('--num_samples', type=int, default=12,
                        help='Number of samples to generate.')
    parser.add_argument('--num_procs', type=int, default=0)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max_iters', type=int, default=999,
                        help='Maximum number of construction ops per graph.')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if not args.output_path:
        sample_and_print(args)
        return

    if args.num_procs < 1:
        sample_and_write_pickle(args, args.output_path, 0)
    else:
        sample_and_write_pickle_multithreaded(args)

if __name__ == '__main__':
    main()
