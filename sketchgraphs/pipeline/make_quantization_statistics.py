"""This script computes statistics required for quantization of continuous parameters.

Many models we use require quantization to process the continuous parameters in the sketch.
This scripts computes the required statistics for quantization of the dataset.

"""

import argparse
import collections
import functools
import gzip
import itertools
import multiprocessing
import pickle
import os

import numpy as np
import tqdm

from sketchgraphs.data import flat_array
from sketchgraphs.data.sequence import EdgeOp
from sketchgraphs.data.sketch import EntityType, ENTITY_TYPE_TO_CLASS
from . import numerical_parameters


_EDGE_PARAMETER_IDS = ('angle', 'length')


def _worker_edges(dataset_path, worker_idx, num_workers, result_queue):
    # Load data
    data = flat_array.load_dictionary_flat(np.load(dataset_path, mmap_mode='r'))
    sequences = data['sequences']

    # Extract sub-sequence for worker
    length_for_worker, num_additional = divmod(len(sequences), num_workers)
    offset = worker_idx * length_for_worker + max(worker_idx, num_additional)
    if worker_idx < num_additional:
        length_for_worker += 1

    seq_indices = range(offset, min((offset+length_for_worker, len(sequences))))

    # Process data
    expression_counters = {
        k: collections.Counter() for k in _EDGE_PARAMETER_IDS
    }

    num_processed = 0

    for seq_idx in seq_indices:
        seq = sequences[seq_idx]

        try:
            for op in seq:
                if not isinstance(op, EdgeOp):
                    continue

                for k in _EDGE_PARAMETER_IDS:
                    if k in op.parameters:
                        value = op.parameters[k]
                        value = numerical_parameters.normalize_expression(value, k)
                        expression_counters[k][value] += 1
        except Exception:
            print('Error processing sequence at index {0}'.format(seq_idx))

        num_processed += 1
        if num_processed > 1000:
            result_queue.put(num_processed)
            num_processed = 0
    result_queue.put(num_processed)

    result_queue.put(expression_counters)


def _worker_node(param_combination, filepath, num_centers, max_values=None):
    label, param_name = param_combination
    sequences = flat_array.load_dictionary_flat(np.load(filepath, mmap_mode='r'))['sequences']

    values = (op.parameters[param_name] for op in itertools.chain.from_iterable(sequences)
              if op.label == label and param_name in op.parameters)

    if max_values is not None:
        values = itertools.islice(values, max_values)

    values = np.array(list(values))
    centers = numerical_parameters.make_quantization(values, num_centers, 'cdf')
    return centers


def process_edges(dataset_path, num_threads):
    print('Checking total sketch dataset size.')
    total_sequences = len(flat_array.load_dictionary_flat(np.load(dataset_path, mmap_mode='r'))['sequences'])

    result_queue = multiprocessing.Queue()

    workers = []

    for worker_idx in range(num_threads):
        workers.append(
            multiprocessing.Process(
                target=_worker_edges,
                args=(dataset_path, worker_idx, num_threads, result_queue)))

    for worker in workers:
        worker.start()

    active_workers = len(workers)

    total_result = {}

    print('Processing sequences for edge statistics')
    with tqdm.tqdm(total=total_sequences) as pbar:
        while active_workers > 0:
            result = result_queue.get()

            if isinstance(result, int):
                pbar.update(result)
                continue

            for k, v  in result.items():
                total_result.setdefault(k, collections.Counter()).update(v)
            active_workers -= 1

    for worker in workers:
        worker.join()

    return total_result


def process_nodes(dataset_path, num_centers, num_threads):
    print('Processing sequences for node statistics')
    label_parameter_combinations = [
        (t, parameter_name)
        for t in (EntityType.Arc, EntityType.Circle, EntityType.Line, EntityType.Point)
        for parameter_name in ENTITY_TYPE_TO_CLASS[t].float_ids
    ]

    pool = multiprocessing.Pool(num_threads)

    all_centers = pool.map(
        functools.partial(
            _worker_node, filepath=dataset_path, num_centers=num_centers, max_values=50000),
        label_parameter_combinations)

    result = {}
    for (t, parameter_name), centers in zip(label_parameter_combinations, all_centers):
        result.setdefault(t, {})[parameter_name] = centers

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input sequence dataset', required=True)
    parser.add_argument('--output', type=str, help='Output dataset path', default='meta.pkl.gz')
    parser.add_argument('--num_threads', type=int, default=0)
    parser.add_argument('--node_num_centers', type=int, default=256)

    args = parser.parse_args()

    num_threads = args.num_threads
    if num_threads is None:
        num_threads = len(os.sched_getaffinity(0))

    edge_results = process_edges(args.input, num_threads)
    node_results = process_nodes(args.input, args.node_num_centers, num_threads)

    print('Saving results in {0}'.format(args.output))
    with gzip.open(args.output, 'wb', compresslevel=9) as f:
        pickle.dump({
            'edge': edge_results,
            'node': node_results
        }, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
