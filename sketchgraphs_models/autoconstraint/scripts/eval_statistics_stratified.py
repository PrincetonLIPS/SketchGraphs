"""Computes edge evaluation statistics stratified by size. """

import argparse
import gzip
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt


def compute_stratified_statistics(statistics, sizes):
    unique_sizes, unique_counts = np.unique(sizes, return_counts=True)

    precision = statistics['precision']
    recall = statistics['recall']

    average_precision = np.empty(len(sizes))
    average_recall = np.empty(len(sizes))
    average_f1 = np.empty(len(sizes))

    sd_precision = np.empty(len(sizes))
    sd_recall = np.empty(len(sizes))
    sd_f1 = np.empty(len(sizes))

    for i, size in enumerate(unique_sizes):
        mask = sizes == size
        pm = precision[mask]
        rm = recall[mask]

        average_precision[i] = pm.mean()
        sd_precision[i] = pm.std() / np.sqrt(len(pm))

        average_recall[i] = rm.mean()
        sd_recall[i] = rm.std() / np.sqrt(len(rm))

        f1 = np.divide(2 * pm * rm, pm + rm, out=np.zeros_like(rm), where=pm + rm != 0)
        average_f1[i] = f1.mean()
        sd_f1[i] = f1.std() / np.sqrt(len(f1))


    return {
        'precision': average_precision,
        'precision_sd': sd_precision,
        'recall': average_recall,
        'recall_sd': sd_recall,
        'f1': average_f1,
        'f1_sd': sd_f1,
        'sizes': unique_sizes,
        'counts': unique_counts,
    }


def boxplot_stratified(statistic, sizes):
    unique_sizes = np.unique(sizes)
    unique_sizes = unique_sizes[(unique_sizes > 3) & (unique_sizes <= 50)]

    return plt.boxplot(
        [statistic[s == sizes] for s in unique_sizes], labels=unique_sizes,
        showfliers=False, showcaps=False, whis=0.0, medianprops={'linewidth': 2.5})


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', required=True)
    parser.add_argument('--output')

    args = parser.parse_args()

    print('Reading input files')

    input_result_path = args.input
    input_basename, input_ext = os.path.splitext(input_result_path)
    if input_ext == '.gz':
        input_basename, _ = os.path.splitext(input_basename)

    input_stat_path = input_basename + '_stat.npz'

    with gzip.open(input_result_path, 'rb') as f:
        ops = pickle.load(f)

    input_stat = np.load(input_stat_path)

    print('Computing statistics')
    stratified_statistics = compute_stratified_statistics(
        input_stat, [len(x['node_ops']) for x in ops])

    if args.output is not None:
        print('Saving result')
        np.savez_compressed(args.output, **stratified_statistics)

    print(stratified_statistics)


if __name__ == '__main__':
    main()
