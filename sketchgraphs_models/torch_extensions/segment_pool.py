import torch

from ._repeat_interleave import repeat_interleave


class SegmentAvgPool1DLoop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, scopes):
        ctx.save_for_backward(scopes)
        ctx.input_length = values.shape[0]
        result = values.new_empty((scopes.shape[0],) + values.shape[1:])

        for i in range(len(scopes)):
            x = values.narrow(0, scopes[i, 0], scopes[i, 1])
            result[i] = x.mean(dim=0)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        scopes, = ctx.saved_tensors
        return segment_avg_pool1d_backward(grad_output, scopes, ctx.input_length), None


_avg_pool_docstring = """
Segmented average pooling.

This function computes an average pool over a set of values, where the
pool is taken over segments defined by scope.

Parameters
----------
values : torch.Tensor
    A 1-dimensional tensor.
scopes : torch.Tensor
    a 2-dimensional integer tensor representing segments.
    Each row of scopes represents a segment, which starts at ``scopes[i, 0]``,
    and has length ``scopes[i, 1]``.

Returns
-------
torch.Tensor
    A tensor representing the average value for each segment.
"""


def segment_avg_pool1d_loop(values, scopes):
    return SegmentAvgPool1DLoop.apply(values, scopes)


def segment_avg_pool1d_scatter(values, scopes):
    import torch_scatter

    lengths = scopes.select(1, 1)
    offsets = lengths.new_zeros(len(lengths) + 1)
    torch.cumsum(lengths, 0, out=offsets[1:])

    return torch_scatter.segment_mean_csr(values, offsets)


segment_avg_pool1d_loop.__docstring__ = _avg_pool_docstring
segment_avg_pool1d_scatter.__docstring__ = _avg_pool_docstring


def segment_avg_pool1d_backward(grad_output, scopes, input_length):
    """ Backward pass for segmented average pooling. """
    scopes_length = scopes.select(1, 1)
    norm_grad = torch.true_divide(grad_output, scopes_length.type_as(grad_output).unsqueeze(1))

    result = grad_output.new_empty([input_length] + list(norm_grad.shape[1:]))
    return repeat_interleave(norm_grad, scopes, dim=0, out=result)


def segment_max_pool1d_backward(grad_output, scopes, max_indices, input_length):
    result = grad_output.new_zeros((input_length, max_indices.shape[1]))

    scopes_offsets = scopes.select(1, 0).unsqueeze(1)
    linear_max_indices = scopes_offsets + max_indices

    result.scatter_(0, linear_max_indices, grad_output)
    return result


class SegmentMaxPool1DLoop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, scopes, return_indices=False):
        result = values.new_empty([scopes.shape[0], values.shape[1]])
        result_idx = values.new_empty([scopes.shape[0], values.shape[1]], dtype=torch.int64)

        for i in range(len(scopes)):
            x = values.narrow(0, scopes[i, 0], scopes[i, 1]).t().unsqueeze(0)
            r, ri = torch.nn.functional.adaptive_max_pool1d(x, 1, return_indices=True)
            r = r.squeeze(0).squeeze(1)
            ri = ri.squeeze(0).squeeze(1)

            result[i] = r
            result_idx[i] = ri

        ctx.save_for_backward(scopes, result_idx)
        ctx.input_length = values.shape[0]
        ctx.mark_non_differentiable(result_idx)

        if return_indices:
            return result, result_idx
        else:
            return result

    @staticmethod
    def backward(ctx, grad_output, *args):
        scopes, result_idx = ctx.saved_tensors
        return segment_max_pool1d_backward(grad_output, scopes, result_idx, ctx.input_length), None, None


def segment_max_pool1d(values: torch.Tensor, scopes: torch.Tensor, return_indices=False) -> torch.Tensor:
    """
    Computes the maximum value in each segment.

    Parameters
    ----------
    values : torch.Tensor
        A 1-dimensional tensor.
    scopes : torch.Tensor
        a 2-dimensional integer tensor representing segments.
        Each row of scopes represents a segment, which starts at ``scopes[i, 0]``,
        and has length ``scopes[i, 1]``.

    Returns
    -------
    torch.Tensor
        A tensor representing the maximum value for each segment.
    """
    return SegmentMaxPool1DLoop.apply(values, scopes, return_indices)


try:
    import torch_scatter

    segment_avg_pool1d = segment_avg_pool1d_scatter
except ImportError:
    segment_avg_pool1d = segment_avg_pool1d_loop


__all__ =  ['segment_avg_pool1d']
