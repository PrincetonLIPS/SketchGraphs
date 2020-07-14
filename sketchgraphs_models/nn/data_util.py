"""Utilities and extensions to work with the torch.utils.data package.
"""

import torch
import torch.utils.data

class MultiEpochSampler(torch.utils.data.Sampler):
    """A sampler wrapper which creates multiple epochs of indices
    from one sampler.

    Due to the way torch dataloaders function, they reset all workers
    when starting a new epoch. This is undesirable due to the high setup
    cost of starting workers.

    This sampler thus wraps an existing batch sampler, and simply repeatedly
    samples several epochs from the wrapped sampler.

    """
    def __init__(self, batch_sampler, num_epochs):
        """Initializes a new instance of `MultiEpochSampler`.

        Parameters
        ----------
        batch_sampler : torch.utils.data.Sampler
            A batch sampler to wrap
        num_epochs : int
            The number of epochs for this sampler.
        """
        #pylint: disable=super-init-not-called
        self._sampler = batch_sampler
        self._epochs = num_epochs

    def __len__(self):
        return len(self._sampler) * self._epochs

    @property
    def batches_per_epoch(self):
        """Get the number of batches per one single epoch of the original sampler."""
        return len(self._sampler)

    def __iter__(self):
        for _ in range(self._epochs):
            for sample in self._sampler:
                yield sample
