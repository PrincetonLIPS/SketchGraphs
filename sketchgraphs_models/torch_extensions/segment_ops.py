import torch
from functools import partial
from ._repeat_interleave import repeat_interleave


def segment_op_python(values, scopes, op):
    if scopes.dim() != 2:
        raise ValueError("Scopes must be two-dimensional.")

    if scopes.shape[1] != 2:
        raise ValueError("Scopes must be of length two in second dimension.")

    output = torch.empty(scopes.shape[0], dtype=values.dtype, device=values.device)

    for i in range(scopes.shape[0]):
        output[i] = op(values.narrow(0, scopes[i, 0], scopes[i, 1]))

    return output


def segment_logsumexp_python(values: torch.Tensor, scopes: torch.Tensor):
    return SegmentLogsumexpPython.apply(values, scopes)


class SegmentLogsumexpPython(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, scopes):
        result = segment_op_python(values, scopes, partial(torch.logsumexp, dim=0))
        ctx.save_for_backward(values, result, scopes)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        values, logsumexp, scopes = ctx.saved_tensors
        lengths = scopes.select(1, 1)

        return segment_logsumexp_backward_python(grad_output, values, logsumexp, lengths), None


def _min_value(dtype):
    if dtype.is_floating_point:
        return torch.finfo(dtype).min
    else:
        return torch.iinfo(dtype).min


def segment_argmax_loop(values, scopes):
    output_values = values.new_empty(scopes.shape[0])
    output_index = scopes.new_empty(scopes.shape[0])

    for i in range(scopes.shape[0]):
        if scopes[i, 1] != 0:
            output_values[i], output_index[i] = torch.max(values.narrow(0, scopes[i, 0], scopes[i, 1]), dim=0)
        else:
            output_values[i] = _min_value(output_values.dtype)
            output_index[i] = -1

    return output_values, output_index


def segment_logsumexp_backward_python(grad_output, values, logsumexp, lengths):
    lengths = lengths.long()
    grad_output_repeat = repeat_interleave(grad_output, lengths, dim=0)
    derivative_repeat = (values - repeat_interleave(logsumexp, lengths, dim=0)).exp_()
    return derivative_repeat.mul_(grad_output_repeat)


class SegmentLogsumexpScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, scopes):
        import torch_scatter

        lengths = scopes.select(1, 1)
        offsets = lengths.new_zeros(len(lengths) + 1)
        torch.cumsum(lengths, 0, out=offsets[1:])

        value_segment_max, _ = torch_scatter.segment_max_csr(values, offsets)
        value_segment_max_expanded = torch.repeat_interleave(value_segment_max, lengths)

        values_exp = values.sub(value_segment_max_expanded).exp_()
        values_sumexp = torch_scatter.segment_sum_csr(values_exp, offsets)

        logsumexp = values_sumexp.log_().add_(value_segment_max)

        ctx.save_for_backward(values, logsumexp, lengths)

        return logsumexp

    @staticmethod
    def backward(ctx, grad_output):
        values, logsumexp, lengths = ctx.saved_tensors

        lse_expand = torch.repeat_interleave(logsumexp, lengths)
        grad_out_expand = torch.repeat_interleave(grad_output, lengths)

        return values.sub(lse_expand).exp_().mul_(grad_out_expand), None


def segment_logsumexp_scatter(values, scopes):
    return SegmentLogsumexpScatter.apply(values, scopes)


def segment_argmax_backward(grad_output, argmax, scopes, input_shape, sparse_grad=True):
    grad_idx = torch.add(argmax, scopes.select(1, 0)).unsqueeze(0)
    grad_input = torch.sparse_coo_tensor(grad_idx, grad_output, size=input_shape)

    if not sparse_grad:
        grad_input = grad_input.to_dense()

    return grad_input


class SegmentArgmaxPython(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, scopes, sparse_grad=True):
        ctx.input_shape = values.shape
        ctx.sparse_grad = sparse_grad

        max_values, argmax = segment_argmax_loop(values, scopes)
        ctx.save_for_backward(argmax, scopes)
        ctx.mark_non_differentiable(argmax)

        return max_values, argmax

    @staticmethod
    def backward(ctx, grad_output, grad_output_index):
        argmax, scopes = ctx.saved_tensors
        grad_values = segment_argmax_backward(grad_output, argmax, scopes, ctx.input_shape, ctx.sparse_grad)
        return grad_values, None, None


def segment_argmax_python(values, scopes, sparse_grad=True):
    return SegmentArgmaxPython.apply(values, scopes, sparse_grad)


def segment_argmax_scatter(values, scopes, sparse_grad=True):
    import torch_scatter

    lengths = scopes.select(1, 1)
    offsets = lengths.new_zeros(len(lengths) + 1)
    torch.cumsum(lengths, 0, out=offsets[1:])

    max_val, argmax = torch_scatter.segment_max_csr(values, offsets)
    max_val = torch.where(lengths != 0, max_val, max_val.new_full((1,), fill_value=torch.finfo(max_val.dtype).min))
    argmax = argmax.sub(offsets[:-1])

    return max_val, argmax


_segment_argmax_docstring = \
"""
Compute the maximum value and location in each segment.

This function computes, for each segment, the maximum value in the segment,
and the offset of the location of the maximum value from the start of the segment.

This function can handle the case where the segment is zero length, in which case
the maximum is given the lowest finite value representable by the type, and the
index is not defined.

Parameters
----------
values : torch.Tensor
    a 1-dimensional `torch.Tensor` representing the values.
scopes : torch.Tensor
    a 2-dimensional integer `torch.Tensor` representing the segments. The ith segment
    has offset ``scopes[i, 0]`` and length ``scopes[i, 1]``.

Returns
-------
torch.Tensor
    A tensor of the same type as `values` representing the maximum in each segment.
torch.Tensor
    An integer tensor representing the location of the maximum in ecah segment.
"""

segment_argmax_scatter.__docstring__ = _segment_argmax_docstring
segment_argmax_python.__docstring__ = _segment_argmax_docstring

_segment_logsumexp_docstring = \
"""
Compute the log-sum-exp in each segment.

This function computes, for each segment, the log-sum-exp value in
a numerically stable fashion.

Parameters
----------
values : torch.Tensor
    A 1-dimensional tensor representing the values.
scopes : torch.Tensor
    A 2-dimensional integer tensor representing the segments. The ith segment
    has offset ``scopes[i, 0]`` and length ``scopes[i, 1]``.

Returns
-------
torch.Tensor
    A tensor representing the log-sum-exp value for each segment.
"""

segment_logsumexp_scatter.__docstring__ = _segment_logsumexp_docstring
segment_logsumexp_python.__docstring__ = _segment_logsumexp_docstring

try:
    import torch_scatter
    segment_logsumexp = segment_logsumexp_scatter
    segment_argmax = segment_argmax_scatter

except ImportError:
    segment_logsumexp = segment_logsumexp_python
    segment_argmax = segment_argmax_python

__all__ = ['segment_logsumexp', 'segment_argmax']
