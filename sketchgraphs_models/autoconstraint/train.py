"""Main module for training auto-constraint model."""

import argparse
import bisect
import datetime
import functools
import json
import os
import time

import torch
import torch.utils.tensorboard
import torch.utils.data

# numpy has come after pytorch due to MKL threading setup
import numpy as np

from sketchgraphs_models.nn.distributed import SingleDeviceDistributedParallel

from sketchgraphs_models import training, distributed_utils
from sketchgraphs_models.autoconstraint import dataset, model as auto_model
from sketchgraphs_models.graph.train import data_loading


_opt_factories = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamax': torch.optim.Adamax,
    'rms': torch.optim.RMSprop
}


def _lr_schedule(epoch, warmup_epochs=5, decay_epochs=None):
    if decay_epochs is None:
        decay_epochs = []

    warmup_factor = min((epoch + 1) / warmup_epochs, 1)
    decay_factor = 0.1 ** (bisect.bisect_right(decay_epochs, epoch))

    return warmup_factor * decay_factor


class AutoconstraintHarness(training.TrainingHarness):
    def __init__(self, model, opt, config_train, config_eval, dist_config, scheduler=None, output_dir=None, profile_enabled=False, additional_model_information=None):
        super(AutoconstraintHarness, self).__init__(model, opt, config_train, config_eval, dist_config)
        self.scheduler = scheduler
        self.output_dir = output_dir
        self.profile_enabled = profile_enabled
        self.additional_model_information = additional_model_information or {}

    def single_step(self, batch, global_step):
        self.opt.zero_grad()
        batch['partner_index'] = training.load_cuda_async(batch['partner_index'], self.config_train.device)

        with torch.autograd.profiler.record_function("forward"):
            readout = self.model(batch)
            losses, accuracy = auto_model.compute_losses(batch, readout)
            total_loss = sum(losses.values())

        if self.model.training:
            with torch.autograd.profiler.record_function("backward"):
                total_loss.backward()

            with torch.autograd.profiler.record_function("opt_update"):
                self.opt.step()

        losses = training.map_structure_flat(losses, lambda x: x.detach())
        losses = auto_model.compute_average_losses(batch, losses)
        avg_loss = total_loss.detach() / batch['graph'].node_counts.shape[0]
        losses['average'] = avg_loss

        return losses, accuracy

    def on_epoch_end(self, epoch, global_step):
        if self.scheduler is not None:
            self.scheduler.step()

            if self.config_train.tb_writer is not None and self.is_leader():
                lr = self.scheduler.get_last_lr()[0]
                self.config_train.tb_writer.add_scalar('learning_rate', lr, global_step)

        if self.is_leader() and self.output_dir is not None and (epoch + 1) % 10 == 0:
            self.log('Saving checkpoint for epoch {}'.format(epoch + 1))
            torch.save({
                    'opt': self.opt.state_dict(),
                    'model': self.model.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    **self.additional_model_information,
                },
                os.path.join(self.output_dir, 'model_state_{0}.pt'.format(epoch + 1)))

    def write_summaries(self, global_step, losses, accuracies, tb_writer):
        if tb_writer is None:
            return

        for k, v in losses.items():
            tb_writer.add_scalar('loss/' + k, v, global_step)

        for k, v in accuracies.items():
            tb_writer.add_scalar('accuracy/' + k, v, global_step)

    def print_statistics(self, loss_acc, accuracy_acc):
        self.log(f'Loss ({loss_acc["average"]:.3f}). Stop ({loss_acc["edge_stop"]:.3f}) Partner ({loss_acc["edge_partner"]:.3f}) Label ({loss_acc["edge_label"]:.3f})')
        self.log(f'Accuracy Stop({accuracy_acc["edge_stop"]:4.1%}) Partner ({accuracy_acc["edge_partner"]:4.1%}) Label ({accuracy_acc["edge_label"]:4.1%})')


def train(node_feature_mapping, dataloader_train, args, output_dir=None, dataloader_eval=None, batches_per_epoch=None, dist_config=None):
    print('Building model.')
    core = auto_model.MODEL_CORES[args['model_core']](args['hidden_size'], node_feature_mapping.feature_dimensions, args['num_prop_rounds'])
    model = auto_model.AutoconstraintModel(core)

    if args['model_state']:
        state = torch.load(args['model_state'], map_location=torch.device('cpu'))

        ## Remove "module." from beginning of keys
        new_state_dict = {}
        for key in state['model']:
            new_state_dict[key[7:]] = state['model'][key]
        state['model'] = new_state_dict
        ##

        model.load_state_dict(state['model'])
        epoch = state['epoch']
        global_step = state['global_step']
    else:
        epoch = 0
        global_step = 0

    if dist_config:
        gpu_id = dist_config.local_rank
        print('Creating model on GPU {0}'.format(gpu_id))
        device = torch.device('cuda', gpu_id)
        # Create parallel device. Note that we need to use find_unused_parameters, as due to the dynamic
        # nature of our computation graph, depending on the available targets in the dataset, not all
        # parameters will have gradients computed for them.
        model = SingleDeviceDistributedParallel(model.to(device), gpu_id, find_unused_parameters=True)
    else:
        device = torch.device('cuda')
        model.to(device)   # Set model device

    print('Model done building.')

    total_batch_size = args['batch_size']
    if dist_config:
        batch_size = total_batch_size // dist_config.world_size
    else:
        batch_size = total_batch_size

    opt = _opt_factories[args['optimizer']](model.parameters(), lr=args['learning_rate'] * total_batch_size / 256)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, functools.partial(_lr_schedule, warmup_epochs=5, decay_epochs=[20, 40]))

    if distributed_utils.is_leader(dist_config):
        tb_writer_main = torch.utils.tensorboard.SummaryWriter(output_dir)
        tb_writer_eval = torch.utils.tensorboard.SummaryWriter(output_dir + '/eval/')
    else:
        tb_writer_main, tb_writer_eval = None, None

    harness = AutoconstraintHarness(
        model, opt,
        training.TrainingConfig(
            dataloader_train,
            tb_writer_main,
            device,
            batch_size,
            batches_per_epoch),
        training.TrainingConfig(
            dataloader_eval,
            tb_writer_eval,
            device,
            batch_size)
        if dataloader_eval is not None
        else None,
        scheduler=scheduler,
        output_dir=output_dir,
        dist_config=dist_config,
        profile_enabled=args['profile'],
        additional_model_information={
            'node_feature_mapping': node_feature_mapping.state_dict(),
            'model_configuration': {
                'embedding_dim': args['hidden_size'],
                'depth': args['num_prop_rounds'],
                'model_core': args['model_core'],
                'name': 'autoconstraint'
            }
        })

    while epoch < args['num_epochs']:
        epoch, global_step = harness.train_epochs(epoch, global_step)

    return model


def get_argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--description', default=None,
                        help='Message describing the current run.')
    parser.add_argument('--output_dir', default='output', help='Directory for output files.')

    parser.add_argument('--dataset_train', required=True,
                        help='Path to training dataset')
    parser.add_argument('--dataset_auxiliary', default=None, help='path to auxiliary dataset containing metadata')
    parser.add_argument('--dataset_test', required=False, default=None,
                        help='Path to validation dataset.')
    parser.add_argument('--model_state', default=None, help='Path to saved model state_dict.')
    parser.add_argument('--num_quantize_length', type=int, default=383, help='number of quantization values for length')
    parser.add_argument('--num_quantize_angle', type=int, default=127, help='number of quantization values for angle')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Training batch size.')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--optimizer', default='adam', choices=list(_opt_factories.keys()))
    parser.add_argument('--hidden_size', type=int, default=384)
    parser.add_argument('--num_prop_rounds', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=60,
                        help='Number of training epochs.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of dataloader workers.')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--world_size', type=int, default=1, help='Number of GPUs to use.')
    parser.add_argument('--profile', action='store_true', help='Whether to produce autograd profiles')
    parser.add_argument('--model_core', type=str, default='bidirectional_recurrent', choices=list(auto_model.MODEL_CORES.keys()))

    return parser


# These keys are converted to absolute paths on save
_ARGS_PATH_KEYS = (
    'output_dir',
    'dataset_train',
    'dataset_auxiliary',
    'dataset_test',
    'model_state'
)


def _feature_dimension(mapping):
    if mapping is None:
        return {}

    return mapping.feature_dimensions


def initialize_datasets(args, distributed_config):
    quantization = {'angle': args['num_quantize_angle'], 'length': args['num_quantize_length']}

    dataset_train_path = args['dataset_train']
    auxiliary_path = args['dataset_auxiliary']

    train_data = data_loading.load_sequences_and_mappings(
        dataset_train_path, auxiliary_path, quantization, edge_features=False)

    ds_train = dataset.AutoconstraintDataset(
        train_data['sequences'], train_data['entity_feature_mapping'], seed=args['seed'])

    batch_size = args['batch_size']
    num_workers = args['num_workers']

    if distributed_config:
        batch_size = batch_size // distributed_config.world_size
        num_workers = num_workers // distributed_config.world_size

    dl_train, batches_per_epoch = data_loading.make_dataloader_train(
        dataset.collate, ds_train, train_data['weights'], batch_size,
        args['num_epochs'], num_workers, distributed_config)

    if args['dataset_test'] is not None:
        raise NotImplementedError('loading testing set not implemented')
    else:
        dl_test = None

    return dl_train, dl_test, batches_per_epoch, train_data['entity_feature_mapping']


def run(args, distributed_config=None):
    """Runs the entire training process according to the given configuration.
    """
    # Set seeds
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])

    for key in _ARGS_PATH_KEYS:
        if args[key] is not None:
            args[key] = os.path.abspath(args[key])

    print('Loading datasets')
    dl_train, dl_test, batches_per_epoch, node_feature_mapping = initialize_datasets(
        args, distributed_config)
    print('Data loaded. Creating output folder.')

    # Derive save_dir
    if distributed_utils.is_leader(distributed_config):
        output_dir = '{}/{}/time_{}'.format(args['output_dir'],
                                            time.strftime('%m%d'),
                                            time.strftime('%H%M%S'))

        os.makedirs(output_dir)

        with open(os.path.join(output_dir, 'args.txt'), 'w') as file_:
            json.dump(args, file_, indent=4)
    else:
        output_dir = None

    print('Starting training.')
    start_time = time.perf_counter()
    _ = train(
        node_feature_mapping,
        dl_train, args,
        output_dir=output_dir, dataloader_eval=dl_test, batches_per_epoch=batches_per_epoch,
        dist_config=distributed_config)
    end_time = time.perf_counter()
    print(f'Done training. Total time: {datetime.timedelta(seconds=end_time - start_time)}.')


def main():
    """Default main function."""
    parser = get_argsparser()
    args = parser.parse_args()

    if args.world_size > 1:
        distributed_utils.train_boostrap_distributed(vars(args), run)
    else:
        run(vars(args))

if __name__ == '__main__':
    main()
