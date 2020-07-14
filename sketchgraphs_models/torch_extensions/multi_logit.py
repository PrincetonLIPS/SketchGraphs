import torch
import numpy as np

from . import segment_ops as _sops


def normalize_values_scopes(values, scopes, batch_size=None, device=None):
    """ Normalizes segmented tensors along with their scopes.

    This function ensures that all the tensors in values and the corresponding
    scopes are normalized. In particular, it takes care of two things:
    - scopes which are None (which denote a segmented tensor of segment size 1
        for each segment) are replaced by their correct scope.
    - values which are not one-dimensional are reshaped to be one-dimensional.
    """
    if len(values) != len(scopes):
        raise ValueError("values and scopes must be lists of the same length.")

    scope_example = next(iter(s for s in scopes if s is not None))

    if batch_size is None:
        batch_size = scope_example.shape[0]

    if device is None:
        device = values[0].device

    dtype = scope_example.dtype

    scopes = [
        s if s is not None else torch.stack([
            torch.arange(0, batch_size, device=device, dtype=dtype),
            torch.ones(batch_size, device=device, dtype=dtype)],
            dim=1)
        for s in scopes]

    for i in range(len(values)):
        if values[i].dim() > 1:
            factor = np.prod(values[i].shape[1:])
            values[i] = values[i].view(-1)
            scopes[i] = scopes[i] * int(factor)

    return values, scopes


def segment_multi_softmax_cross_entropy_loop(logits, scopes, label):
    loss = torch.empty(label.shape[0], device=logits[0].device)

    for i in range(label.shape[0]):
        instance_logits = torch.cat([
            torch.narrow(tl, 0, s[i, 0], s[i, 1]) for tl, s in zip(logits, scopes)])

        loss[i] = torch.nn.functional.cross_entropy(
            instance_logits.unsqueeze(0), label[i].unsqueeze(0).long())

    return loss


def select_label_multi_segment_loop(values, scopes, label):
    """ Selects a value from the given values for each label.

    This function considers a list of ragged tensors (represented by values and scopes),
    a selects the nth element of the concatenation of these tensors across batches
    where n is given by the label.

    This function performs this is a naive fashion using a loop
    across the batch size.
    """
    batch_size = label.shape[0]

    result = torch.empty(
        batch_size, dtype=values[0].dtype, device=values[0].device)

    for i in range(batch_size):
        values_batch = torch.cat([
            torch.narrow(v, 0, s[i, 0], s[i, 1])
            for v, s in zip(values, scopes)])

        result[i] = values_batch[label[i]]

    return result


def _segment_across_offsets(segment_lengths):
    segment_across_offsets = torch.zeros_like(segment_lengths)
    torch.cumsum(segment_lengths[:, :-1], dim=1, out=segment_across_offsets[:, 1:])
    return segment_across_offsets


def select_label_multi_segment_python(values, scopes, label):
    """ Selects a value from the given values for each label by index. """
    batch_size = label.shape[0]
    label = label.unsqueeze(1).long()

    all_scopes = torch.stack(scopes, dim=2)
    segment_lengths = all_scopes.select(1, 1)
    segment_batch_offsets = all_scopes.select(1, 0)

    segment_across_offsets = _segment_across_offsets(segment_lengths)

    indices_across = torch.sum(label >= segment_across_offsets, dim=1) - 1
    indices = label - segment_across_offsets

    indices = torch.min(indices, segment_lengths - 1)
    indices = indices.clamp_(min=0)

    indices_linear = indices.add_(segment_batch_offsets)
    max_linear_index = indices_linear.new_tensor([v.shape[0] - 1 for v in values])

    indices_linear = torch.min(indices_linear, max_linear_index.unsqueeze(0))

    all_selected_list = []

    for i, v in enumerate(values):
        all_selected_list.append(
            v.index_select(0, indices_linear.select(1, i)))

    all_selected = torch.stack(all_selected_list, dim=1)

    return all_selected.gather(1, indices_across.unsqueeze(1)).squeeze(1)


select_label_multi_segment = select_label_multi_segment_python


def segment_multi_softmax_cross_entropy(logits, scopes, label):
    """ Segmented multi-task softmax cross-entropy loss.

    This function computes a softmax cross-entropy loss on logits which
    are across different tensors. Each of these tensors also represent ragged
    tensors, as the number of actions can vary per batch instance.

    Parameters
    ----------
    logits: list of one-dimensional tensors, representing the logits of the actions.
    scopes: a list of two-dimensional tensors, of size N x 2, where N is the batch size.
        The first column represents the offset of the segment corresponding to the given
        batch, and the second column represents the length of the segment corresponding to
        the given batch.

    label: a integer tensor representing the label of the true action. This label is computed
        in a linear fashion, ie taking into account each action from all logit tensors.
    """
    all_norm = torch.stack([
        _sops.segment_logsumexp(logit, scope) for logit, scope in zip(logits, scopes)],
        dim=0)

    log_label_prob = select_label_multi_segment(logits, scopes, label)
    norm = all_norm.logsumexp(dim=0)
    return norm - log_label_prob


def segment_multi_softmax_coarse_fine(logits, scopes, label_coarse, label_fine, return_coarse_prob=False):
    """ Segmented multi-task softmax cross-entropy loss with coarse-to-fine loss.

    This function is similar to `segment_multi_softmax_cross_entropy`, but also
    computes a coarse loss which determines whether the predicted coarse action
    type is correct. Each coarse action is represented by a set of logits.

    Parameters
    ----------
    logits: an iterable of tensors representing the logits for each task.
    scopes: an iterable of tensor representing the scopes defining each segments.
    label_coarse: a tensor representing the coarse label.
    label_fine: a tensor representing the fine label.
    return_coarse_prob: if True, also return the coarse action probabilities.
    """
    all_norm = torch.stack([
        _sops.segment_logsumexp(logit, scope) for logit, scope in zip(logits, scopes)],
        dim=0)

    log_fine_label_prob = select_label_multi_segment(logits, scopes, label_fine)
    log_coarse_label_prob = all_norm.gather(0, label_coarse.unsqueeze(0)).squeeze(0)

    global_norm = all_norm.logsumexp(dim=0)

    fine_loss = torch.mean(global_norm - log_fine_label_prob)
    coarse_loss = torch.mean(global_norm - log_coarse_label_prob)

    if return_coarse_prob:
        return coarse_loss, fine_loss, all_norm.t()
    else:
        return coarse_loss, fine_loss


def segment_multi_argmax_loop(values, scopes):
    locations = []
    maximums = []

    for i in range(scopes[0].shape[0]):
        logits = torch.cat([
            torch.narrow(tl, 0, s[i, 0], s[i, 1])
            for tl, s in zip(values, scopes) if s[i, 1] != 0])

        v, l = torch.max(logits, dim=0)
        maximums.append(v)
        locations.append(l)

    return torch.stack(maximums), torch.stack(locations)


def segment_multi_argmax(values, scopes):
    maximums = []
    locations = []

    for v, s in zip(values, scopes):
        m, l = _sops.segment_argmax(v, s)
        maximums.append(m)
        locations.append(l)

    maximums = torch.stack(maximums, dim=1)
    locations = torch.stack(locations, dim=1)

    segment_lengths = torch.stack(scopes, dim=2).select(1, 1)
    locations = locations.add_(_segment_across_offsets(segment_lengths))

    values, indices = torch.max(maximums, dim=1)
    return values, locations.gather(1, indices.unsqueeze(1)).squeeze(1)
