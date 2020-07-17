"""This script is responsible for creating the main sketch dataset from the JSON files.

Some basic filtering is applied in order to exclude empty sketches. However, as this dataset
is intended to capture the original data, further filtering is left to scripts such as
`sketchgraphs.pipeline.make_sequence_dataset` which process dataset for learning.

"""

import argparse
import collections
import glob
import gzip
import itertools
import json
import multiprocessing as mp
import tarfile
import traceback
import os

import numpy as np
import tqdm
import zstandard as zstd

from sketchgraphs.data.sketch import Sketch
from sketchgraphs.data import flat_array


def _load_json(path):
    open_ = gzip.open if path.endswith('gz') else open
    with open_(path) as fh:
        return json.load(fh)


def filter_sketch(sketch: Sketch):
    """Basic filtering which excludes empty sketches, or sketches with no constraints."""
    return len(sketch.constraints) == 0 or len(sketch.entities) == 0


def parse_sketch_id(filename):
    basename = os.path.basename(filename)
    while '.' in basename:
        basename, _ = os.path.splitext(basename)

    document_id, part_id = basename.split('_')
    return document_id, int(part_id)


def load_json_tarball(path):
    """Loads a json tarball as an iterable of sketches.

    Parameters
    ----------
    path : str
        A path to the location of a single shard

    Returns
    -------
    iterable of `Sketch`
        An iterable of `Sketch` representing all the sketches present in the tarball.
    """
    with open(path, 'rb') as base_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(base_file) as tarball:
            with tarfile.open(fileobj=tarball, mode='r|') as directory:
                while True:
                    json_file = directory.next()
                    if json_file is None:
                        break

                    if not json_file.isfile():
                        continue

                    document_id, part_id = parse_sketch_id(json_file.name)
                    data = directory.extractfile(json_file).read()
                    if len(data) == 0:
                        # skip empty files
                        continue

                    try:
                        sketches_json = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise ValueError('Error decoding JSON for document {0} part {1}.'.format(document_id, part_id))
                    for i, sketch_json in enumerate(sketches_json):
                        yield (document_id, part_id, i), Sketch.from_fs_json(sketch_json)



def _worker(paths_queue, processed_sketches, max_sketches, sketch_counter):
    num_filtered = 0
    num_invalid = 0

    while max_sketches is None or sketch_counter.value < max_sketches:
        paths = paths_queue.get()

        if paths is None:
            break

        sketches = []

        for path in paths:
            sketch_list = _load_json(path)

            for sketch_json in sketch_list:
                try:
                    sketch = Sketch.from_fs_json(sketch_json)
                except Exception as err:
                    num_invalid += 1
                    print('Error processing sketch in file {0}'.format(path))
                    traceback.print_exception(type(err), err, err.__traceback__)

                if filter_sketch(sketch):
                    num_filtered += 1
                    continue

                sketches.append(sketch)

        offsets, data = flat_array.raw_list_flat(sketches)

        processed_sketches.put((offsets, data))

        with sketch_counter.get_lock():
            sketch_counter.value += len(sketches)

    processed_sketches.put({
        'num_filtered': num_filtered,
        'num_invalid': num_invalid
    })


def process(paths, threads, max_sketches=None):
    path_queue = mp.Queue()
    sketch_queue = mp.Queue()
    sketch_counter = mp.Value('q', 0)

    # Enqueue all the objects
    print('Enqueueing files to process.')
    paths_it = iter(paths)
    while True:
        path_chunk = list(itertools.islice(paths_it, 128))
        if len(path_chunk) == 0:
            break

        path_queue.put_nowait(path_chunk)

    workers = []

    for _ in range(threads or mp.cpu_count()):
        workers.append(
            mp.Process(
                target=_worker,
                args=(path_queue, sketch_queue, max_sketches, sketch_counter)))

    for worker in workers:
        path_queue.put_nowait(None)
        worker.start()

    active_workers = len(workers)

    offsets_arrays = []
    data_arrays = []

    statistics = collections.Counter()

    # Read-in data
    with tqdm.tqdm(total=len(paths)) as pbar:
        while active_workers > 0:
            result = sketch_queue.get()

            if isinstance(result, dict):
                statistics += collections.Counter(result)
                active_workers -= 1
                continue

            offsets, data = result
            offsets_arrays.append(offsets)
            data_arrays.append(data)

            pbar.update(128)

    # Finalize workers
    for worker in workers:
        worker.join()

    # Merge final flat array
    all_offsets, all_data = flat_array.merge_raw_list(offsets_arrays, data_arrays)
    total_sketches = len(all_offsets) - 1
    del offsets_arrays
    del data_arrays

    # Pack as required
    flat_data = flat_array.pack_list_flat(all_offsets, all_data)
    del all_offsets
    del all_data

    print('Done processing data.\nProcessed sketches: {0}'.format(total_sketches))
    print('Filtered sketches: {0}'.format(statistics['num_filtered']))
    print('Invalid sketches: {0}'.format(statistics['num_invalid']))
    return flat_data


def gather_sorted_paths(patterns):
    if isinstance(patterns, str):
        patterns = [patterns]
    out = []
    for pattern in patterns:
        out.extend(glob.glob(pattern))
    out.sort()
    return out


def main():
    parser = argparse.ArgumentParser(description='Process json files to create sketch dataset')
    parser.add_argument('--glob_pattern', required=True, action='append',
                        help='Glob pattern(s) for json / json.gz files.')
    parser.add_argument('--output_path', required=True, help='Path for output file.')
    parser.add_argument('--max_files', type=int, help='Max number of json files to consider.')
    parser.add_argument('--max_sketches', type=int, help='Maximum number of sketches to consider.')
    parser.add_argument('--num_threads', type=int, default=0, help='Number of multiprocessing workers.')

    args = parser.parse_args()

    print('Globbing for sketch files to include.')
    paths = gather_sorted_paths(args.glob_pattern)
    print('Found %i files.' % len(paths))
    if args.max_files is not None:
        paths = paths[:args.max_files]

    result = process(paths, args.num_threads, args.max_sketches)

    print('Saving data to {0}'.format(args.output_path))
    np.save(args.output_path, result)


if __name__ == '__main__':
    main()
