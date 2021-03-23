"""Utilities for model training.

This module implements the main training harness, which is shared by both the graph
and autoconstrain model.

"""

import abc
import collections
import dataclasses
import datetime
import functools
import numbers
import time
import typing

import numpy as np
import torch
import torch.utils.tensorboard

from sketchgraphs_models import distributed_utils

_scalar_types = (
    torch.Tensor, np.ndarray, numbers.Number, np.int32, np.int64
)


def map_structure_flat(structure, function, scalar_types=None):
    """Utility function for mapping a function over an arbitrary structure,
    maintaining the structure.

    Parameters
    ----------
    structure : object
        An arbitrary nested structure
    function : function
        A function to apply to each leaf of the structure
    scalar_types : Tuple, optional
        If not None, a tuple of types considered scalar types over which the
        function is directly applied

    Returns
    -------
    object
        A structure with each element modified.

    Raises
    ------
    ValueError
        If the type is not a scalar type and is not decomposable, an exception is raised.
    """
    map_structure_fn = functools.partial(map_structure_flat, function=function, scalar_types=scalar_types)

    if scalar_types is None:
        scalar_types = _scalar_types

    if structure is None:
        return None

    if isinstance(structure, scalar_types):
        return function(structure)

    if hasattr(structure, '_make'):
        return structure._make(map(map_structure_fn, structure))

    if dataclasses.is_dataclass(structure):
        return dataclasses.replace(structure, **map_structure_fn(vars(structure)))

    if isinstance(structure, collections.OrderedDict):
        return collections.OrderedDict([(k, map_structure_fn(v)) for k, v in structure.items()])

    if isinstance(structure, collections.abc.Mapping):
        return type(structure)([(k, map_structure_fn(v)) for k, v in structure.items()])

    if isinstance(structure, collections.abc.Sequence):
        return type(structure)([map_structure_fn(v) for v in structure])

    raise ValueError('Unsupported structure type {0}'.format(type(structure)))


def load_cuda_async(batch, device=None):
    """Loads a structured batch recursively onto the given torch device."""
    if device is not None and device.type != "cuda":
        return batch

    load_cuda_async_device = functools.partial(load_cuda_async, device=device)

    if batch is None:
        return None
    elif isinstance(batch, torch.Tensor) or isinstance(batch, torch.nn.utils.rnn.PackedSequence):
        return batch.to(device=device, non_blocking=False)
    elif hasattr(batch, '_make'):
        # Check for namedtuple
        return batch._make(map(load_cuda_async_device, batch))
    elif dataclasses.is_dataclass(batch):
        # Check for @dataclass
        return dataclasses.replace(batch, **load_cuda_async_device(vars(batch)))
    elif isinstance(batch, collections.OrderedDict):
        return collections.OrderedDict([(k, load_cuda_async_device(v)) for k, v in batch.items()])
    elif isinstance(batch, collections.abc.Mapping):
        return {k: load_cuda_async_device(v) for k, v in batch.items()}
    elif isinstance(batch, collections.abc.Sequence):
        return [load_cuda_async_device(v) for v in batch]
    elif isinstance(batch, (int, np.int32, np.int64)):
        return batch
    else:
        raise ValueError("Unsupported batch collection type {0}.".format(type(batch)))


def _accumulate(losses, acc):
    for k, v in losses.items():
        if v is None:
            continue

        if isinstance(v, dict):
            _accumulate(v, acc.setdefault(k, {}))
        else:
            acc.setdefault(k, v.new_zeros(v.shape)).add_(v.detach())


class TrainingConfig(typing.NamedTuple):
    """Named tuple holding configuration for training a given model."""
    dataloader: torch.utils.data.DataLoader
    tb_writer: typing.Optional[torch.utils.tensorboard.SummaryWriter]
    device: torch.device

    batch_size: int
    batches_per_epoch: typing.Optional[int] = None


class TrainingHarness(abc.ABC):
    """This class implements the main training loop."""
    def __init__(self, model, opt, config_train: TrainingConfig, config_eval: TrainingConfig = None,
                 dist_config: distributed_utils.DistributedTrainingInfo = None):
        """Creates a new harness for the given model.

        Parameters
        ----------
        model : torch.nn.Module
            The torch model to train.
        opt : torch.optim.Optimizer
            The optimizer to use during training
        config_train : TrainingConfig
            The configuration to use for training
        config_eval : TrainingConfig, optional
            The configuration to use for evaluation
        dist_config : DistributedTrainingInfo, optional
            The configuration used for distributed training
        """
        self.model = model
        self.opt = opt
        self.config_train = config_train
        self.config_eval = config_eval
        self.dist_config = dist_config


    @abc.abstractmethod
    def single_step(self, batch, global_step):
        """Implements a single step of the model evaluation / training.

        Parameters
        ----------
        batch : dict
            Input batch from the dataloader
        global_step : int
            Global step for this batch

        Returns
        -------
        losses : dict
            Dictionary of computed losses
        accuracy : dict
            Dictionary of computed accuracy
        """

    def is_leader(self):
        return distributed_utils.is_leader(self.dist_config)

    def on_epoch_end(self, epoch, global_step):
        """This function is called at the end of each epoch."""
        pass

    def write_summaries(self, global_step, losses, accuracies, tb_writer):
        pass

    def print_statistics(self, loss_acc, accuracy_acc):
        pass

    def reset_statistics(self):
        pass

    def log(self, *args):
        if self.is_leader():
            print(*args)

    def train_epochs(self, start_epoch=0, global_step=0):
        """Trains the model for a single iteration over the dataloader.

        Note that usually, a single iteration over a dataloader represents a single epoch.
        However, because starting a new epoch is very expensive for the dataloader, we instead
        allow dataloaders to iterate over multiple epochs at a time.

        Parameters
        ----------
        start_epoch : int
            The current epoch before training
        global_step : int
            The current global step before training

        Returns
        -------
        epoch : int
            The current epoch after training
        global_step : int
            The current global step after training
        """
        last_time = time.perf_counter()

        loss_acc = {}
        accuracy_acc = {}

        batch_idx = 0  # Mini-batch index within each epoch.
        epoch = start_epoch
        epoch_start_time = None

        batches_per_epoch = (self.config_train.batches_per_epoch or len(self.config_train.dataloader.batch_sampler))
        log_every_n = min(50, batches_per_epoch)
        total_batch_size = self.config_train.batch_size
        if self.dist_config:
            total_batch_size *= self.dist_config.world_size


        self.model.train()

        for j, batch in enumerate(self.config_train.dataloader):
            if j % batches_per_epoch == 0:
                # new epoch initialization
                epoch += 1
                self.log(f'Starting epoch #{epoch}')
                epoch_start_time = time.perf_counter()

                # reset batch counters
                batch_idx = 0
                loss_acc = {}
                accuracy_acc = {}
                last_time = epoch_start_time

            batch = load_cuda_async(batch, device=self.config_train.device)
            losses, accuracy = self.single_step(batch, global_step)

            _accumulate(losses, loss_acc)
            _accumulate(accuracy, accuracy_acc)

            global_step += total_batch_size

            if (batch_idx + 1) % log_every_n == 0:
                if self.is_leader():
                    self.write_summaries(global_step, losses, accuracy, self.config_train.tb_writer)

                current_time = time.perf_counter()
                elapsed = current_time - last_time
                last_time = current_time
                graph_per_second = log_every_n * total_batch_size / elapsed

                self.log(f'Epoch {epoch}. Batch {batch_idx + 1}. {graph_per_second:.2f} graphs / s')

                if self.is_leader():
                    loss_acc = map_structure_flat(loss_acc, lambda x: x / float(log_every_n))
                    accuracy_acc = map_structure_flat(accuracy_acc, lambda x: x / float(log_every_n))
                    self.print_statistics(loss_acc, accuracy_acc)

                self.reset_statistics()
                loss_acc = {}
                accuracy_acc = {}

            if (j + 1) % batches_per_epoch == 0:
                # epoch end
                self.on_epoch_end(epoch, global_step)

                current_time = time.perf_counter()
                self.log(f'Done with epoch #{epoch}. '
                         f'Took {datetime.timedelta(seconds=current_time - epoch_start_time)}\n')
                self.run_holdout_eval(epoch, global_step)
                self.model.train()

            batch_idx += 1

        if (j + 1) % batches_per_epoch != 0:
            print('Warning: incomplete epoch')

        return epoch, global_step


    def run_holdout_eval(self, epoch, global_step):
        """Runs the holdout evaluation process.

        Parameters
        ----------
        epoch : int
            The current epoch of training
        global_step : int
            The current global step of training
        """
        if self.config_eval is None:
            self.log('Skipping holdout evaluation as no evaluation dataset specified.')
            return

        self.log('Running holdout eval...')
        loss_acc = collections.OrderedDict()
        accuracy_acc = collections.OrderedDict()
        self.reset_statistics()

        self.model.eval()

        idx = 0
        for idx, batch in enumerate(self.config_eval.dataloader):
            batch = load_cuda_async(batch, device=self.config_eval.device)
            with torch.no_grad():
                losses, accuracy = self.single_step(batch, global_step)
            _accumulate(losses, loss_acc)
            _accumulate(accuracy, accuracy_acc)

        num_batches = idx + 1

        loss_acc = map_structure_flat(loss_acc, lambda x: x / num_batches)
        accuracy_acc = map_structure_flat(accuracy_acc, lambda x: x / num_batches)

        if self.is_leader():
            self.log(f'Eval for epoch={epoch}, global_step={global_step}:')
            self.print_statistics(loss_acc, accuracy_acc)
            self.log()
            self.write_summaries(global_step, loss_acc, accuracy_acc, self.config_eval.tb_writer)
