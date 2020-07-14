"""Utility functions for distributed (multi-gpu) training.
"""

import math
import socket
import typing

import torch
import torch.distributed

from functools import partial


class DistributedTrainingInfo(typing.NamedTuple):
    sync_url: str
    world_size: int
    rank: int
    local_rank: int

    def __bool__(self):
        return self.world_size > 1



def is_leader(config: DistributedTrainingInfo):
    """Tests whether the current process is the leader for distributed training."""
    return config is None or config.rank == 0


def train_boostrap_distributed(parameters, train):
    world_size = parameters.get('world_size', 1)

    if world_size == 1:
        # Single-node training, nothing to do.
        parameters['rank'] = 0
        return train(parameters)

    parameters['rank'] = -1
    from torch.multiprocessing import spawn

    sync_url = parameters.get('sync_url')
    if sync_url is None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        port = s.getsockname()[1]
        s.close()
        sync_url = f'tcp://127.0.0.1:{port}'
        parameters['sync_url'] = sync_url
        print(f'Using URL bootstrap at {sync_url}')

    spawn(partial(train_distributed, train=train), nprocs=parameters['world_size'], args=(parameters,))


def initialize_distributed(config: DistributedTrainingInfo):
    from torch import distributed as dist

    dist.init_process_group(
        'nccl', init_method=config.sync_url,
        world_size=config.world_size, rank=config.rank)

    torch.cuda.set_device(config.local_rank)


def get_distributed_config(parameters, local_rank=None):
    world_size = parameters.get('world_size', 1)

    if world_size == 1:
        return DistributedTrainingInfo('', 1, 0, 0)

    sync_url = parameters['sync_url']

    rank = local_rank

    return DistributedTrainingInfo(sync_url, world_size, rank, local_rank)


def train_distributed(local_rank, parameters, train):
    config = get_distributed_config(parameters, local_rank)
    initialize_distributed(config)
    train(parameters, config)


class DistributedSampler(torch.utils.data.Sampler):
    """Utility class which adapts a sampler into a distributed sampler which only
    samples a subset of the underlying sampler, according to a division by rank.
    """
    def __init__(self, sampler, num_replicas=None, rank=None):
        if num_replicas is not None:
            if not torch.distributed.is_available():
                raise RuntimeError("DistributedSampler requires torch distributed to be available")
            num_replicas = torch.distributed.get_world_size()

        if rank is not None:
            if not torch.distributed.is_available():
                raise RuntimeError("DistributedSampler requires torch distributed to be available")

        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(sampler) / self.num_replicas))
        self.total_size = self.num_replicas * self.num_samples

    def __iter__(self):
        indices = list(self.sampler)
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples
