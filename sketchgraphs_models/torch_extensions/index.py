import torch


def segment_triu_indices_loop(scopes, offset=0, dtype=torch.long, device='cpu'):
    indices = torch.cat([
        torch.triu_indices(
            n, n, offset, dtype=dtype, device=device).t() + o for (o, n) in scopes
    ], dim=0)

    return indices


def segment_cartesian_product_loop(values_a, values_b, scopes_a, scopes_b):
    result = []

    for sa, sb in zip(scopes_a, scopes_b):
        va = values_a.narrow(0, sa[0], sa[1]).unsqueeze(1).expand((-1, sb[1], -1))
        vb = values_b.narrow(0, sb[0], sb[1]).unsqueeze(0).expand((sa[1], -1, -1))

        values_cat = torch.cat((va, vb), dim=-1)
        result.append(values_cat.flatten(end_dim=1))

    return torch.cat(result, dim=0)


segment_triu_indices = segment_triu_indices_loop
segment_cartesian_product = segment_cartesian_product_loop
