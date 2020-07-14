import torch


def segment_index_add_python(values, scopes, indices, out=None):
    if out is None:
        out = values.new_zeros([scopes.shape[0]] + list(values.shape[1:]))

    scopes = scopes.long()

    values_dup = values.index_select(0, indices)
    idx_global = torch.repeat_interleave(scopes[:, 1])
    out.index_add_(0, idx_global, values_dup)

    return out


segment_index_add = segment_index_add_python
