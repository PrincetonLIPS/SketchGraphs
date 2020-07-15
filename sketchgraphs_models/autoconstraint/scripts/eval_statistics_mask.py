"""Evaluate statistics from the masks used applied to samples in the data."""

import argparse
import functools

import numpy as np
import torch
from torch import multiprocessing
import tqdm

from sketchgraphs.data import flat_array
from sketchgraphs.data.sequence import NodeOp
from sketchgraphs_models.autoconstraint.eval import MASK_FUNCTIONS


def count_valid_choices_by_step(node_ops, mask_function):
    if node_ops[-1].label == 'Stop':
        node_ops = node_ops[:-1]

    masks = mask_function(node_ops)

    valid_choices = np.ones(len(node_ops) - 1, dtype=np.int)

    for i in range(1, len(node_ops)):
        mask_offset = i * (i + 1) // 2
        valid_choices[i - 1] = (torch.narrow(masks, 0, mask_offset, i + 1) == 0).int().sum().item() + 1

    return valid_choices


def total_valid_choices(seq, mask_function):
    node_ops = [op for op in seq if isinstance(op, NodeOp)]

    valid_choices = count_valid_choices_by_step(node_ops, mask_function)

    if len(valid_choices) > 0:
        average_choices = valid_choices.mean()
        average_entropy = np.log2(valid_choices).mean()
    else:
        # These values will be ignored
        average_choices = 0
        average_entropy = 0

    return average_choices, average_entropy, len(valid_choices)


def uniform_valid_perplexity(seqs, mask_function, num_workers=0):
    if num_workers is None or num_workers > 0:
        pool = multiprocessing.Pool(num_workers)
        map_fn = functools.partial(pool.imap, chunksize=8)
    else:
        map_fn = map

    valid_choices_fn = functools.partial(total_valid_choices, mask_function=mask_function)

    average_choices = np.empty(len(seqs))
    average_entropy = np.empty(len(seqs))
    seq_length = np.empty(len(seqs), dtype=np.int)

    for i, (choices, entropy, length) in enumerate(tqdm.tqdm(map_fn(valid_choices_fn, seqs), total=len(seqs))):
        average_choices[i] = choices
        average_entropy[i] = entropy
        seq_length[i] = length

    return {
        'choices': average_choices,
        'entropy': average_entropy,
        'sequence_length': seq_length
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--mask', choices=list(MASK_FUNCTIONS.keys()), default='node_type')
    parser.add_argument('--output')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=16)

    args = parser.parse_args()

    print('Reading data')
    seqs = flat_array.load_dictionary_flat(np.load(args.input, mmap_mode='r'))['sequences']
    seqs.share_memory_()

    if args.limit is not None:
        seqs = seqs[:args.limit]

    print('Computing statistics')
    result = uniform_valid_perplexity(seqs, MASK_FUNCTIONS[args.mask], args.num_workers)

    if args.output is not None:
        print('Saving results')
        np.savez_compressed(args.output, **result)

    choices = np.average(result['choices'], weights=result['sequence_length'])
    entropy = np.average(result['entropy'], weights=result['sequence_length'])
    print('Average choices: {:.3f}'.format(choices))
    print('Average entropy: {:.3f}'.format(entropy))

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
