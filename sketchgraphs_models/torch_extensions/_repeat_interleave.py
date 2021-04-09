import torch

def _ensure_repeats(repeats_or_scopes: torch.Tensor):
    if repeats_or_scopes.dim() == 2:
        return repeats_or_scopes.select(1, 1)
    else:
        return repeats_or_scopes


def repeat_interleave_out(values, repeats_or_scope, dim, out):
    index = torch.repeat_interleave(_ensure_repeats(repeats_or_scope))

    if dim is None:
        values = values.flatten()
        dim = 0

    return torch.index_select(values, dim, index, out=out)


def repeat_interleave(values, repeats_or_scope, dim=None, out=None, out_length=None):
    """ Extended `repeat_interleave` with the ability to pass in pre-computed information
    to reduce CPU-GPU communication when the repeats tensor resides on GPU.

    In particular, this function can make use of a scope parameter instead of simply repeats,
    which is a two-dimensional tensor with two columns, one corresponding to the offsets
    of each segment, and the second one corresponding to the length (i.e. the number of repeats).

    Parameters
    ----------
    values : torch.Tensor
        An array of values to repeat.
    repeats_or_scope : torch.Tensor
        Either a one-dimensional array of repeats, or a two-dimensional array
        representing the offset and length (which is referred to as scope).
    dim : int, optional
        The dimension of values along which to repeat.
    out : torch.Tensor, optional
        if not None, a tensor into which to produce the output. Note that this is not differentiable.
    out_length : int, optional
        if not None, the length of the output along the repeated dimension.
    Returns
    -------
    A new tensor, containing values repeated the appropriate amount of times along the
    given dimension.
    """
    if out is not None:
        return repeat_interleave_out(values, repeats_or_scope, dim, out)
    return torch.repeat_interleave(values, repeats_or_scope, dim)
