""" Utility functions for computing specific nn functions. """

import torch
from sketchgraphs_models.torch_extensions import segment_logsumexp, segment_avg_pool1d
from sketchgraphs_models.torch_extensions.segment_ops import segment_argmax


def segmented_cross_entropy(logits: torch.Tensor, target: torch.Tensor, scopes: torch.Tensor) -> torch.Tensor:
    """Segmented cross-entropy loss.

    Computes the cross-entropy loss from unscaled `logits` for a segmented problem.

    Parameters
    ----------
    logits : torch.Tensor
        unscaled logits by segment
    target : torch.Tensor
        tensor of length `n_segments`, representing the index of the true label
        for each segment.
    scopes : tonch.Tensor
        tensor of shape `[n_segments, 2]`, representing the segments as `(start, length)`.

    Returns
    -------
    torch.Tensor
        A tensor of length `n_segments` representing the cross-entropy loss
        at each segment.
    """
    input_logsumexp = segment_logsumexp(logits, scopes)
    return logits.index_select(0, target) - input_logsumexp


def segmented_multinomial(logits, scopes, generator=None):
    """Segmented multinomial sample.

    Parameters
    ----------
    logits : torch.Tensor
        unscaled logits by segment
    scopes : torch.Tensor
        tensor of shape `[n_segments, 2]` representing the segments as `(start, length)`.
    generator : torch.Generator, optional
        PRNG for sampling

    Returns
    -------
    torch.Tensor
        A tensor of length `n_segments` representing the sampled values.
    """
    output = logits.new_empty([scopes.shape[0]], dtype=torch.int64)

    logits = logits.detach()

    for i in range(scopes.shape[0]):
        segment_logits = logits.narrow(0, scopes[i, 0], scopes[i, 1])
        torch.multinomial(
            torch.nn.functional.softmax(segment_logits, dim=0), num_samples=1,
            generator=generator,
            out=output[i:i])

    return output


def segmented_multinomial_extended(logits, scopes, generator=None, also_return_probs=False):
    """Segmented multinomial sample with implicit element.

    Parameters
    ----------
    logits : torch.Tensor
        logits for explicit outcomes by segment
    scopes : torch.Tensor
        tensor of shape `[n_segments, 2]` representing the segments as `(start, length)`.
    generator : torch.Generator, optional
        PRNG for sampling
    also_return_probs: bool, optional
        If true, returns tuple including log-likelihood of the sample.

    Returns
    -------
    torch.Tensor
        A tensor of length `n_segments` representing the sampled values.
    """

    output = logits.new_empty([scopes.shape[0]], dtype=torch.int64)

    logits = logits.detach()

    for i in range(scopes.shape[0]):
        segment_logits = logits.new_zeros([scopes[i, 1] + 1])
        segment_logits[:-1] = logits.narrow(0, scopes[i, 0], scopes[i, 1])

        dist = torch.nn.functional.softmax(segment_logits, dim=0)
        torch.multinomial(dist, num_samples=1, generator=generator,
                          out=output[i:i])
    if also_return_probs:
        return output, dist[output]
    return output
