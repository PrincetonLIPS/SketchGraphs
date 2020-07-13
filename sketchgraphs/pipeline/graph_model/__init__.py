"""This module provides the data pipeline to convert from the stored sequence format to
a format suitable for training our graph-based deep learning model. In particular, it features
a number of important helpers for quantizing numerical features, and representing the graph in
terms of tensors.

"""

from . import _graph_info
from ._graph_info import *

__all__ = _graph_info.__all__
