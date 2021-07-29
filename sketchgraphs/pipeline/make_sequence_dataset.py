"""This script is responsible for creating sequence datasets from the main sketch dataset.

This script performs many important pipeline steps to prepare the data into a format that is suitable
for training deep learning models. In addition, it is also configurable with a number of filtering options,
in order to ensure that learning is performed on adequate data.

"""

import argparse
import collections
import enum
import functools
import itertools
import multiprocessing
import traceback
import os

import numpy as np
import tqdm

from sketchgraphs.data import flat_array, sequence
from sketchgraphs.data.sketch import Sketch, ConstraintType, EntityType

from sketchgraphs.pipeline.make_sketch_dataset import load_json_tarball, filter_sketch as base_filter_sketch


from .numerical_parameters import normalize_expression


class FilterReason(enum.Enum):
    Accepted = 0
    TooManyEntities = 1
    TooFewEntities = 2
    TooManyConstraints = 3
    SequenceTooShort = 4
    InvalidEntityType = 5
    InvalidConstraintType = 6
    Empty = 7


def filter_sketch(sketch: Sketch, config: dict) -> FilterReason:
    if base_filter_sketch(sketch):
        return FilterReason.Empty

    if len(sketch.entities) > config['max_entities']:
        return FilterReason.TooManyEntities

    if len(sketch.entities) < config['min_entities']:
        return FilterReason.TooFewEntities

    if len(sketch.constraints) > config['max_constraints']:
        return FilterReason.TooManyConstraints

    if len(sketch.entities) + len(sketch.constraints) < config['min_sequence_length']:
        return FilterReason.SequenceTooShort

    if any(entity.type in config['rejected_entity_types'] for entity in sketch.entities.values()):
        return FilterReason.InvalidEntityType

    if any(constraint.type in config['rejected_constraint_types'] for constraint in sketch.constraints.values()):
        return FilterReason.InvalidConstraintType

    return FilterReason.Accepted


def make_default_filter_config(min_size=0, max_size=2 ** 30, min_sequence_length=0):
    return {
        'min_entities': min_size,
        'max_entities': max_size,
        'max_constraints': 4 * max_size,
        'min_sequence_length': min_sequence_length,
        'rejected_entity_types': (EntityType.Conic, EntityType.Ellipse, EntityType.Spline, EntityType.Unknown),
        'rejected_constraint_types': (ConstraintType.Circular_Pattern, ConstraintType.Linear_Pattern,
                                      ConstraintType.Mirror, ConstraintType.Projected)
    }


def make_empty_filter_config(*_):
    return {
        'min_entities': 0,
        'max_entities': 2 ** 32,
        'max_constraints': 2 ** 32,
        'min_sequence_length': 0,
        'rejected_entity_types': (),
        'rejected_constraint_types': (),
    }


def _normalize_constraint_parameters(seq):
    for op in seq:
        if not isinstance(op, sequence.EdgeOp):
            continue

        for param in ('angle', 'length'):
            if param not in op.parameters:
                continue
            value = op.parameters[param]
            op.parameters[param] = normalize_expression(value, param) or value


def _get_sketch_iterable(config):
    if isinstance(config['dataset_path'], list) or config['dataset_path'].endswith('.tar.zst'):
        return _sketch_iterable_from_json_dataset(config)
    else:
        raise NotImplementedError('Can only load from json tarballs currently')


def _sketch_iterable_from_json_dataset(config):
    num_workers = config['num_workers']
    dataset_path = config['dataset_path']
    worker_idx = config['worker_idx']

    if not isinstance(dataset_path, list):
        dataset_path = [dataset_path]

    splits = np.array_split(dataset_path, num_workers)
    files = splits[worker_idx]

    return itertools.chain.from_iterable(
        load_json_tarball(p) for p in files
    )


def _worker(config, processed_sequences, filter_function):
    worker_idx = config['worker_idx']
    chunk_size = config['chunk_size']

    sketches = _get_sketch_iterable(config)

    filtered_reasons = collections.Counter()
    num_invalid = 0

    sequences = []
    sketch_ids = []
    count_in_chunk = 0
    chunk_index = 0
    num_filtered_in_chunk = 0

    for sketch_id, sketch in sketches:
        filter_reason = filter_function(sketch)

        if filter_reason != FilterReason.Accepted:
            filtered_reasons[filter_reason] += 1
            num_filtered_in_chunk += 1
            continue

        try:
            seq = sequence.sketch_to_sequence(sketch)
            _normalize_constraint_parameters(seq)
            sequences.append(seq)
            sketch_ids.append(sketch_id)
            count_in_chunk += 1
        except Exception as err:
            num_invalid += 1
            print('Error processing sketch {2} in document {0} part {1}.'.format(*sketch_id))
            traceback.print_exception(type(err), err, err.__traceback__)

        if count_in_chunk >= chunk_size:
            sequence_lengths = np.array([len(seq) for seq in sequences], dtype=np.int64)
            sketch_ids = np.array(
                sketch_ids, dtype=[('document_id', 'S24'), ('part_idx', '<i4'), ('sketch_idx', '<i4')])

            offsets, sequence_data = flat_array.raw_list_flat(sequences)

            processed_sequences.put(
                ((worker_idx, chunk_index), offsets, sequence_data, sequence_lengths,
                 sketch_ids, num_filtered_in_chunk))

            sequences = []
            sketch_ids = []
            count_in_chunk = 0
            chunk_index += 1
            num_filtered_in_chunk = 0

    # Send final batch of data
    sequence_lengths = np.array([len(seq) for seq in sequences], dtype=np.int64)
    sketch_ids = np.array(sketch_ids, dtype=[('document_id', 'S24'), ('part_idx', '<i4'), ('sketch_idx', '<i4')])

    offsets, sequence_data = flat_array.raw_list_flat(sequences)
    processed_sequences.put(
        ((worker_idx, chunk_index), offsets, sequence_data, sequence_lengths, sketch_ids, num_filtered_in_chunk))

    processed_sequences.put({
        'filtered': filtered_reasons,
        'errors': num_invalid
    })


def process(dataset_path, threads, filter_function, total_sketches=None):
    sequence_queue = multiprocessing.Queue()
    workers = []

    chunk_size = 128

    common_config = {
        'dataset_path': dataset_path,
        'num_workers': threads,
        'chunk_size': chunk_size
    }

    for worker_idx in range(threads):
        config = {
            'worker_idx': worker_idx,
            **common_config
        }

        workers.append(
            multiprocessing.Process(
                target=_worker,
                args=(config, sequence_queue, filter_function)))

    for worker in workers:
        worker.start()

    active_workers = len(workers)
    sequence_results = []
    filtered_statistics = collections.Counter()
    total_errors = 0

    print('Processing sketches')
    with tqdm.tqdm(total=total_sketches, smoothing=0.01) as pbar:
        while active_workers > 0:
            result = sequence_queue.get()

            if isinstance(result, dict):
                filtered_statistics += result['filtered']
                total_errors += result['errors']
                active_workers -= 1
                continue

            sequence_results.append(result)
            pbar.update(chunk_size + result[-1])

    for worker in workers:
        worker.join()

    print('Sorting processed sequences')
    sequence_results.sort(key=lambda x: x[0])

    print('Merging processed sequences')
    flat_data = flat_array.pack_list_flat(*flat_array.merge_raw_list(
        [x[1] for x in sequence_results],
        [x[2] for x in sequence_results]))

    all_sequence_lengths = np.concatenate([x[3] for x in sequence_results])
    all_sketch_ids = np.concatenate([x[4] for x in sequence_results])

    flat_dict = flat_array.pack_dictionary_flat({
        'sequences': flat_data,
        'sequence_lengths': all_sequence_lengths,
        'sketch_ids': all_sketch_ids,
    })

    print('Done processing sequences')
    print('Total accepted: {0}'.format(len(all_sequence_lengths)))
    print('Total filtered: {0}'.format(sum(filtered_statistics.values())))
    print('Total errors: {0}'.format(total_errors))
    print('Filtered by type: {0}'.format(filtered_statistics))

    return flat_dict


BASE_FILTER_FACTORIES = {
    'empty': make_empty_filter_config,
    'default': make_default_filter_config
}


def main():
    parser = argparse.ArgumentParser(description='Process a sketch dataset into sequences')
    parser.add_argument('--input', type=str, required=True, nargs='+', help='Path to input file')
    parser.add_argument('--output', type=str, required=True, help='Path to output file')

    parser.add_argument('--max_size', type=int, default=16,
                        help='Max number of entities allowed in sketch (6 is 75th percentile).')
    parser.add_argument('--min_entities', type=int, default=0,
                        help='Min number of entities allowed in sketch.')
    parser.add_argument(
        '--min_sequence_length', type=int, default=0,
        help='Minimum number of sequence operations (entity + constraints) in sketch.')
    parser.add_argument(
        '--num_threads', type=int, default=None, help='Number of multiprocessing workers.')
    parser.add_argument(
        '--base_filter', choices=list(BASE_FILTER_FACTORIES.keys()), default='default')

    parser.add_argument(
        '--total_sketches', type=int, default=14930928)

    args = parser.parse_args()

    filter_factory = BASE_FILTER_FACTORIES[args.base_filter]
    filter_config = filter_factory(args.min_entities, args.max_size, args.min_sequence_length)

    num_threads = args.num_threads
    if num_threads is None:
        num_threads = len(os.sched_getaffinity(0))

    filter_function = functools.partial(filter_sketch, config=filter_config)

    result = process(args.input, num_threads, filter_function, total_sketches=args.total_sketches)

    print('Saving result at {0}'.format(args.output))
    np.save(args.output, result, allow_pickle=False)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
