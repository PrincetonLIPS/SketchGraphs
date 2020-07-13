"""This script is responsible for creating sequence datasets from the main sketch dataset.

This script performs many important pipeline steps to prepare the data into a format that is suitable
for training deep learning models. In addition, it is also configurable with a number of filtering options,
in order to ensure that learning is performed on adequate data.

"""

import argparse
import collections
import enum
import multiprocessing
import traceback

import numpy as np
import tqdm

from sketchgraphs.data.sketch import Sketch, ConstraintType, EntityType
from sketchgraphs.data import flat_array, sequence

from .numerical_parameters import normalize_expression


class FilterReason(enum.Enum):
    Accepted = 0
    TooManyEntities = 1
    TooFewEntities = 2
    TooManyConstraints = 3
    SequenceTooShort = 4
    InvalidEntityType = 5
    InvalidConstraintType = 6


def filter_sketch(sketch: Sketch, config):
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


def _normalize_constraint_parameters(seq):
    for op in seq:
        if not isinstance(op, sequence.EdgeOp):
            continue

        for param in ('angle', 'length'):
            if param not in op.parameters:
                continue
            value = op.parameters[param]
            op.parameters[param] = normalize_expression(value, param) or value


def _worker(config, processed_sequences, filter_config):
    seed = config['seed']
    dataset_path = config['dataset_path']
    worker_idx = config['worker_idx']
    num_workers = config['num_workers']
    chunk_size = config['chunk_size']

    sketch_array = flat_array.load_flat_array(dataset_path)

    if seed is not None:
        rng = np.random.Generator(np.random.Philox(seed))
        indices = rng.permutation(len(sketch_array))
    else:
        indices = range(len(sketch_array))

    filtered_reasons = collections.Counter()
    num_invalid = 0

    for chunk_idx in range(worker_idx, len(indices) // chunk_size, num_workers):
        sequences = []

        for i in indices[chunk_idx * chunk_size:chunk_idx * chunk_size + chunk_size]:
            sketch = sketch_array[i]
            filter_reason = filter_sketch(sketch, filter_config)

            if filter_reason != FilterReason.Accepted:
                filtered_reasons[filter_reason] += 1
                continue

            try:
                seq = sequence.sketch_to_sequence(sketch)
                _normalize_constraint_parameters(seq)
                sequences.append(seq)
            except Exception as err:
                num_invalid += 1
                print('Error processing sketch at sketch index {0}'.format(i))
                traceback.print_exception(type(err), err, err.__traceback__)

        sequence_lengths = np.array([len(seq) for seq in sequences], dtype=np.int64)
        offsets, sequence_data = flat_array.raw_list_flat(sequences)

        processed_sequences.put((chunk_idx, offsets, sequence_data, sequence_lengths))

    processed_sequences.put({
        'filtered': filtered_reasons,
        'errors': num_invalid
    })


def process(dataset_path, seed, threads, filter_config):
    sequence_queue = multiprocessing.Queue()
    workers = []

    print('Checking total sketch dataset size.')
    total_sketches = len(flat_array.load_flat_array(dataset_path))
    chunk_size = 512

    common_config = {
        'seed': seed,
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
                args=(config, sequence_queue, filter_config)))

    for worker in workers:
        worker.start()

    active_workers = len(workers)
    sequence_results = []
    filtered_statistics = collections.Counter()
    total_errors = 0

    print('Processing sketches')
    with tqdm.tqdm(total=total_sketches) as pbar:
        while active_workers > 0:
            result = sequence_queue.get()

            if isinstance(result, dict):
                filtered_statistics += result['filtered']
                total_errors += result['errors']
                active_workers -= 1
                continue

            sequence_results.append(result)
            pbar.update(chunk_size)

    for worker in workers:
        worker.join()

    print('Sorting processed sequences')
    sequence_results.sort(key=lambda x: x[0])

    print('Merging processed sequences')
    flat_data = flat_array.pack_list_flat(*flat_array.merge_raw_list(
        [x[1] for x in sequence_results],
        [x[2] for x in sequence_results]))

    all_sequence_lengths = np.concatenate([x[3] for x in sequence_results])

    flat_dict = flat_array.pack_dictionary_flat({
        'sequences': flat_data,
        'sequence_lengths': all_sequence_lengths
    })

    print('Done processing sequences')
    print('Total accepted: {0}'.format(len(flat_data)))
    print('Total filtered: {0}'.format(sum(filtered_statistics.values())))
    print('Total errors: {0}'.format(total_errors))
    print('Filtered by type: {0}'.format(filtered_statistics))

    return flat_dict


def main():
    parser = argparse.ArgumentParser(description='Process a sketch dataset into sequences')
    parser.add_argument('--input', type=str, required=True, help='Path to input file')
    parser.add_argument('--output', type=str, required=True, help='Path to output file')

    parser.add_argument('--max_size', type=int, default=16,
                        help='Max number of entities allowed in sketch (6 is 75th percentile).')
    parser.add_argument('--min_entities', type=int, default=0,
                        help='Min number of entities allowed in sketch.')
    parser.add_argument('--min_sequence_length', type=int, default=0,
                        help='Minimum number of sequence operations (entity + constraints) in sketch.')
    parser.add_argument('--num_threads', type=int, default=0,
                        help='Number of multiprocessing workers.')
    parser.add_argument('--seed', type=int, default=None, help='Seed used to shuffle dataset order, if None, do not shuffle')

    args = parser.parse_args()

    filter_config = make_default_filter_config(args.min_entities, args.max_size, args.min_sequence_length)
    result = process(args.input, args.seed, args.num_threads, filter_config)

    print('Saving result at {0}'.format(args.output))
    np.save(args.output, result, allow_pickle=False)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
