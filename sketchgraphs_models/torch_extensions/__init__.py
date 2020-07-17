""" Pytorch extensions for our models.

This module provides some important extensions that work with segmented data, which is
used extensively in our models due to the nature of graph batching. This modules provides
a pure python implementation, although can leverage a C++ extension if present. This
enables speed-up of over 2x (on GPU) when it is present, so it is recommended to compile the C++
extensions if possible.

"""

from ._repeat_interleave import repeat_interleave
from .segment_ops import segment_logsumexp
from .segment_pool import segment_avg_pool1d, segment_max_pool1d

from .index import segment_triu_indices, segment_cartesian_product

__all__ = ['repeat_interleave', 'segment_logsumexp', 'segment_avg_pool1d', 'segment_max_pool1d',
           'segment_triu_indices', 'segment_cartesian_product']
