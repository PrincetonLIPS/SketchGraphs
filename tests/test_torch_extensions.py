import pytest
import torch
import numpy as np

from scipy.special import logsumexp

from sketchgraphs_models.torch_extensions import _repeat_interleave, segment_ops, segment_pool


def test_repeat_python():
    x = np.random.randn(40).reshape(4, 10)
    times = [2, 5, 0, 1]

    expected = np.repeat(x, times, axis=0)
    result = _repeat_interleave.repeat_interleave(torch.tensor(x), torch.tensor(times), 0)

    assert np.allclose(result.numpy(), expected)


def test_segment_logsumexp_python():
    x = np.random.randn(40)
    lengths = [5, 10, 6, 4, 15]
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1])))

    scopes = np.stack((offsets, lengths), axis=1)

    expected = np.array([logsumexp(x[s[0]:s[0] + s[1]]) for s in scopes])
    result = segment_ops.segment_logsumexp_python(torch.tensor(x), torch.tensor(scopes))

    assert np.allclose(result, expected)


def test_segment_logsumexp_python_grad():
    x = np.random.randn(40)

    lengths = [5, 10, 6, 4, 15]
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1])))

    scopes = np.stack((offsets, lengths), axis=1)

    torch.autograd.gradcheck(
        segment_ops.segment_logsumexp_python,
        (torch.tensor(x, requires_grad=True), torch.tensor(scopes)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_segment_logsumexp_scatter(device):
    x = np.random.randn(40)
    lengths = [0, 5, 10, 6, 4, 15, 0]
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1])))

    scopes = np.stack((offsets, lengths), axis=1).astype(np.int64)

    expected = np.array([logsumexp(x[s[0]:s[0] + s[1]]) if s[1] != 0 else -np.inf for s in scopes])

    result = segment_ops.segment_logsumexp_scatter(torch.tensor(x, device=device), torch.tensor(scopes, device=device))

    assert np.allclose(result.cpu().numpy(), expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_segment_logsumexp_scatter_grad(device):
    x = np.random.randn(40)

    lengths = [5, 10, 6, 4, 15]
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1])))

    scopes = np.stack((offsets, lengths), axis=1).astype(np.int64)

    torch.autograd.gradcheck(
        segment_ops.segment_logsumexp_scatter,
        (torch.tensor(x, requires_grad=True, device=device), torch.tensor(scopes, device=device)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_segment_logsumexp_scatter_grad_full(device):
    x = np.random.randn(20)

    scopes = torch.tensor([[0, 20]], dtype=torch.int64, device=device)

    torch.autograd.gradcheck(
        segment_ops.segment_logsumexp_scatter,
        (torch.tensor(x, requires_grad=True, device=device), scopes))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_segment_argmax(device):
    x = np.random.randn(40)

    lengths = np.array([0, 5, 10, 6, 4, 15, 0])
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1])))

    scopes = np.stack((offsets, lengths), axis=1).astype(np.int64)

    x = torch.tensor(x, device=device)
    scopes = torch.tensor(scopes, device=device)

    expected_values, expected_index = segment_ops.segment_argmax_python(x, scopes)
    result_values, result_index = segment_ops.segment_argmax_scatter(x, scopes)

    result_values = result_values.cpu().numpy()
    expected_values = expected_values.cpu().numpy()
    result_index = result_index.cpu().numpy()
    expected_index = expected_index.cpu().numpy()

    assert np.allclose(result_values, expected_values)
    assert np.allclose(result_index[lengths > 0], expected_index[lengths > 0])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_segment_argmax_backward(device):
    x = np.random.randn(40)

    lengths = [5, 10, 6, 4, 15]
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1])))

    scopes = np.stack((offsets, lengths), axis=1).astype(np.int64)

    torch.autograd.gradcheck(
        segment_ops.segment_argmax_scatter,
        (torch.tensor(x, requires_grad=True, device=device),
         torch.tensor(scopes, device=device),
         False))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_segment_pool(device):
    x = np.random.randn(40)

    lengths = [5, 10, 6, 4, 15]
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1])))

    scopes = np.stack((offsets, lengths), axis=1).astype(np.int64)

    x = torch.tensor(x, device=device)
    scopes = torch.tensor(scopes, device=device)

    expected_values = segment_pool.segment_avg_pool1d_loop(x, scopes)
    result_values = segment_pool.segment_avg_pool1d_scatter(x, scopes)

    assert torch.allclose(expected_values, result_values)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_segment_pool_2d(device):
    x = np.random.randn(40, 5)

    lengths = [5, 10, 6, 4, 15]
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1])))

    scopes = np.stack((offsets, lengths), axis=1).astype(np.int64)

    x = torch.tensor(x, device=device)
    scopes = torch.tensor(scopes, device=device)

    expected_values = segment_pool.segment_avg_pool1d_loop(x, scopes)
    result_values = segment_pool.segment_avg_pool1d_scatter(x, scopes)

    assert torch.allclose(expected_values, result_values)
