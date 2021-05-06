"""Utility to benchmark data loading speed."""

import argparse
import time


from sketchgraphs_models.graph.train.data_loading import initialize_datasets


def time_iterator(iterator, batch_size, total_time_seconds=None):
    start_time = time.perf_counter()
    last_time = start_time
    num_batch_processed = 0

    for _ in iterator:
        num_batch_processed += 1
        current_time = time.perf_counter()
        elapsed = current_time - last_time
        if elapsed > 5:
            print('Processed {0} elements in {1:.2f} seconds. ({2:.2f} / second )'.format(
                num_batch_processed * batch_size, elapsed, num_batch_processed * batch_size / elapsed ))
            last_time = time.perf_counter()
            num_batch_processed = 0

        if total_time_seconds is not None and current_time - start_time > total_time_seconds:
            break


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_train', required=True,
                        help='Pickle dataset for train data.')
    parser.add_argument('--dataset_auxiliary', default=None, help='path to auxiliary dataset containing metadata')
    parser.add_argument('--num_quantize_length', type=int, default=383, help='number of quantization values for length')
    parser.add_argument('--num_quantize_angle', type=int, default=127, help='number of quantization values for angle')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of dataloader workers.')
    parser.add_argument('--disable_edge_features', action='store_true',
                        help='Disable using and predicting edge features')

    parser.add_argument('--max_time', type=float, default=300)

    args = vars(parser.parse_args())
    args['dataset_test'] = None
    args['disable_entity_features'] = True

    print('Loading dataset')
    dataloader, _, batches_per_epoch, _, _ = initialize_datasets(args)

    print('Benchmarking dataloader')
    time_iterator(dataloader, args['batch_size'], args['max_time'])


if __name__ == '__main__':
    main()

