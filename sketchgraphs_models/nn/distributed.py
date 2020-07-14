"""Utility modules for distributed and parallel training. """

import torch

class SingleDeviceDistributedParallel(torch.nn.parallel.distributed.DistributedDataParallel):
    """This module implements a module similar to `DistributedDataParallel`, but it accepts
    inputs of any shape, and only supports a single device per instance.
    """
    def __init__(self, module, device_id, find_unused_parameters=False):
        super(SingleDeviceDistributedParallel, self).__init__(
            module, [device_id], find_unused_parameters=find_unused_parameters)

    def forward(self, *inputs, **kwargs):
        if self.require_forward_param_sync:
            self._sync_params()

        output = self.module(*inputs, **kwargs)

        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            self.require_forward_param_sync = True

            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(torch.nn.parallel.distributed._find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])

        return output

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.module.load_state_dict(state_dict, strict)
