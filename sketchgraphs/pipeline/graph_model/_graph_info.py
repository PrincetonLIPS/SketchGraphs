"""This module provides the basic structure for representing a graph for use in our learning models.

This module provides utilities to work with graphs and batched graphs represented as
thin wrappers over collections of torch tensors.

"""

import dataclasses
import typing
import torch


def offsets_from_counts(counts):
    """Creates a tensor of offsets from the given tensor of counts

    Parameters
    ----------
    counts : torch.Tensor
        1-d tensor representing the counts in a ragged array

    Returns
    -------
    torch.Tensor
        A 1-d tensor representing the offsets in a ragged array.
        Its length is one plus the length of the input `counts` array.
    """
    if isinstance(counts, torch.Tensor):
        device = counts.device
    else:
        device = None
    counts = torch.as_tensor(counts, dtype=torch.int64, device=device)
    offsets = counts.new_zeros(counts.shape[0] + 1)
    torch.cumsum(counts, 0, out=offsets[1:])
    return offsets


def scopes_from_offsets(offsets):
    """Compute scopes (specification used by the segmented operations) from array of offsets

    Parameters
    ----------
    offsets: torch.Tensor
        Tensor of length `n_segments + 1` representing the offsets of each segment

    Returns
    -------
    torch.Tensor
        A tensor of shape `[n_segment, 2]` representing segment specification in the form `(start, length)`.
    """
    return torch.stack((offsets[:-1], offsets[1:] - offsets[:-1]), dim=-1)


@dataclasses.dataclass
class SparseFeatureBatch:
    """This class represents a batch of sparse features.

    Attributes
    ----------
    index
        A tensor of shape [n] representing the indices of the features that are present
    value
        A tensor of shape [n, ...] representing the values of the features at the given indices
    """
    index: torch.Tensor
    value: torch.Tensor

    def __init__(self, index: torch.Tensor, value: torch.Tensor):
        """Initializes a new instance of `SparseFeatureBatch`.

        Parameters
        ----------
        index
            The locations recorded in this instance
        value
            The features that this instance represents
        batch_offsets
            If not None, delineates batches in this feature. Otherwise, it is assumed that there is
            only a single batch.
        """
        self.index = index
        self.value = value

    @property
    def indices(self):
        return self.index

    @property
    def values(self):
        return self.value

    @staticmethod
    def merge(features, offsets):
        """Merges a list of sparse feature batches into a ragged array.

        This function merges a list of SparseFeatureBatch assuming that the underlying
        indexing is done in ragged fashion.
        """
        indices = []
        values = []

        for feature, offset in zip(features, offsets):
            indices.append(feature.index + offset)
            values.append(feature.value)

        return SparseFeatureBatch(
            torch.cat(indices, dim=0) if indices else torch.empty([0], dtype=torch.int64),
            torch.cat(values, dim=0) if values else torch.empty([0], dtype=torch.int64))

    @staticmethod
    def pack(features, batch_offsets):
        """Packs a list of sparse feature batches into swizzled array.

        This function packs a list of SparseFeatureBatch assuming that the underlying
        indexing is done in swizzled fashion, similar to `torch.nn.utils.rnn.PackedSequence`.
        """
        indices = []
        values = []

        for i, feature in enumerate(features):
            indices.append(i + batch_offsets[feature.index])
            values.append(feature.value)

        return SparseFeatureBatch(
            torch.cat(indices, dim=0) if indices else torch.empty([0], dtype=torch.int64),
            torch.cat(values, dim=0) if values else torch.empty([0], dtype=torch.int64))

    def share_memory_(self):
        self.index.share_memory_()
        self.value.share_memory_()
        return self

    def apply(self, fn):
        """Creates a new `SparseFeatureBatch` with the given fields transformed according to the given function.
        """
        return type(self)(fn(self.index), fn(self.value))


@dataclasses.dataclass
class GraphInfo:
    """This class wraps an edge list and associated metadata as a view over a batch of graphs.

    Attributes
    ----------
    incidence
        A tensor of shape `[2, n_edges]` representing the edges in the graph.
    node_counts
        A tensor of length `n_graphs` representing the number of nodes in each graph in the batch.
    node_offsets
        A tensor of length `n_graphs + 1` representing the node offsets of each graph in the batch.
    edge_counts
        A tensor of length `n_graphs` representing the number of edges in each graph in the batch.
    edge_offsets
        A tensor of length `n_graphs + 1` representing the edge offsets of each graph in the batch.
    node_features
        A tensor of length `n_nodes` representing the features associated with each node.
    edge_features
        A tensor of length `n_edges` representing the features associated with each edge.
    sparse_edge_features
        A dictionary containing instances `SparseEdgeFeature`, representing features which are only present
        on a subset of the edges.
    """
    incidence: torch.Tensor
    node_counts: torch.Tensor
    node_offsets: torch.Tensor
    edge_counts: torch.Tensor
    edge_offsets: torch.Tensor
    node_features: typing.Optional[torch.Tensor] = None
    edge_features: typing.Optional[torch.Tensor] = None
    sparse_node_features: typing.Dict[str, SparseFeatureBatch] = None
    sparse_edge_features: typing.Dict[str, SparseFeatureBatch] = None


    def _getslice(self, start, stop, step):
        if step != 1:
            raise ValueError('can only obtain contiguous slices')

        node_counts = self.node_counts[start:stop]
        edge_counts = self.edge_counts[start:stop]

        node_start = self.node_offsets[start]
        edge_start = self.edge_offsets[start]

        node_stop = self.node_offsets[stop]
        edge_stop = self.edge_offsets[stop]

        node_offsets = self.node_offsets[start:stop + 1] - node_start
        edge_offsets = self.edge_offsets[start:stop + 1] - edge_start

        slice_incidence = self.incidence[:, edge_start:edge_stop] - node_start

        slice_node_data = self.node_features[node_start:node_stop] if self.node_features is not None else None
        slice_edge_data = self.edge_features[edge_start:edge_stop] if self.edge_features is not None else None

        if self.sparse_edge_features is not None:
            sparse_edge_features = {
                k: v.slice(start, stop, edge_start)
                for k, v in self.sparse_edge_features.items()
            }
        else:
            sparse_edge_features = None

        if self.sparse_node_features is not None:
            sparse_node_features = {
                k: v.slice(start, stop, node_start)
                for k, v in self.sparse_node_features.items()
            }
        else:
            sparse_node_features = None

        return GraphInfo(
            slice_incidence, node_counts, node_offsets, edge_counts, edge_offsets,
            slice_node_data, slice_edge_data, sparse_node_features, sparse_edge_features)


    def gather(self, indices):
        """Creates a sub-batch of graphes from the given batches in this instance of `GraphInfo`.

        Parameters
        ----------
        indices : iterable of int
            A list of indices from which to collect the graphs.

        Returns
        -------
        GraphInfo
            A batch of graphs containing the selected graphs.
        """
        slices = [self[int(i)] for i in indices]
        return GraphInfo.merge(*slices)


    def __getitem__(self, idx):
        """Extracts a graph or a range of graphs from the batch.

        Returns
        -------
        GraphInfo
            A `GraphInfo` object containing the data from the associated slice.

        Raises
        ------
        IndexError
            If the index is out of range
        """
        try:
            iter_idx = iter(idx)
        except TypeError:
            iter_idx = None

        if iter_idx is not None:
            return self.gather(iter_idx)

        if not isinstance(idx, slice):
            if idx > len(self):
                raise IndexError()
            idx = slice(idx, idx + 1)

        return self._getslice(*idx.indices(len(self)))

    def __len__(self):
        return len(self.node_counts)


    @staticmethod
    def from_single_graph(incidence, node_features=None, edge_features=None, num_nodes=None,
                          sparse_node_features=None, sparse_edge_features=None):
        """Creates a graph info from a single graph with the given edge list.

        Parameters
        ----------
        incidence : torch.Tensor
            A `torch.Tensor` of dimension `[2, num_edges]` representing the edges in the graph as an edge list.
        num_nodes : int, optional
            If not None, an integer representing the number of nodes in the graph. Otherwise, this is inferred
            as the largest node index referred to in the `incidence` list.
        node_features : torch.Tensor, optional
            If not None, a tensor representing features associated with the nodes.
        edge_features : torch.Tensor, optional
            If not None, a tensor representing features associated with the edges.
        sparse_node_features: dictionary, optional
            If not None, a dictionary of `SparseFeatures`, representing sparse features associated with nodes.
        sparse_edge_features : dictionary, optional
            If not None, a dictionary of `SparseFeatures`, representing sparse features associated
            with the edges.

        Returns
        -------
        GraphInfo
            A graph representing the given data.
        """
        incidence = torch.as_tensor(incidence)

        if num_nodes is None:
            if node_features is None:
                num_nodes = torch.max(incidence)
            else:
                num_nodes = node_features.shape[0]

        if node_features is not None and num_nodes != node_features.shape[0]:
            raise ValueError('node_info should have length the number of nodes.')

        node_counts = torch.as_tensor(num_nodes).reshape([1])
        node_offsets = incidence.new_tensor([0, num_nodes])
        edge_counts = incidence.new_tensor([incidence.shape[1]])
        edge_offsets = incidence.new_tensor([0, incidence.shape[1]])

        return GraphInfo(incidence, node_counts, node_offsets, edge_counts, edge_offsets,
                         node_features, edge_features, sparse_node_features, sparse_edge_features)


    @staticmethod
    def merge(*graphs):
        """Merges several graph batches into a single batch.

        Parameters
        ----------
        *graphs
            a collection of `GraphInfo` to merge.

        Returns
        -------
        GraphInfo
            A graph containing all the combined graphs.
        """
        incidences = []
        node_counts = []
        edge_counts = []

        node_features = []
        edge_features = []
        sparse_node_features = []
        sparse_edge_features = []

        total_nodes = 0
        total_edges = 0

        batch_node_offsets = [0]
        batch_edge_offsets = [0]

        for graph in graphs:
            incidences.append(graph.incidence + total_nodes)
            node_counts.append(graph.node_counts)
            edge_counts.append(graph.edge_counts)

            total_nodes += int(graph.node_offsets[-1])
            total_edges += int(graph.edge_offsets[-1])

            node_features.append(graph.node_features)
            edge_features.append(graph.edge_features)

            sparse_node_features.append(graph.sparse_node_features)
            sparse_edge_features.append(graph.sparse_edge_features)

            batch_node_offsets.append(total_nodes)
            batch_edge_offsets.append(total_edges)

        incidence = torch.cat(incidences, dim=-1) if incidences else torch.empty([2, 0], dtype=torch.int64)

        node_counts = torch.cat(node_counts) if node_counts else torch.empty([0], dtype=torch.int64)
        edge_counts = torch.cat(edge_counts) if edge_counts else torch.empty([0], dtype=torch.int64)

        node_offsets = offsets_from_counts(node_counts)
        edge_offsets = offsets_from_counts(edge_counts)

        if node_features and all(v is not None for v in node_features):
            node_features = torch.cat(node_features, dim=0)
        else:
            node_features = None

        if edge_features and all(v is not None for v in edge_features):
            edge_features = torch.cat(edge_features, dim=0)
        else:
            edge_features = None

        if sparse_node_features and all(v is not None for v in sparse_node_features):
            all_features = sparse_node_features
            sparse_node_features = {}

            for k in all_features[0]:
                sparse_node_features[k] = SparseFeatureBatch.merge(
                    [f[k] for f in all_features], batch_node_offsets[:-1])
        else:
            sparse_node_features = None

        if sparse_edge_features and all(v is not None for v in sparse_edge_features):
            all_features = sparse_edge_features
            sparse_edge_features = {}

            for k in all_features[0]:
                sparse_edge_features[k] = SparseFeatureBatch.merge(
                    [f[k] for f in all_features], batch_edge_offsets[:-1])
        else:
            sparse_edge_features = None

        return GraphInfo(
            incidence, node_counts, node_offsets,
            edge_counts, edge_offsets, node_features, edge_features, sparse_node_features, sparse_edge_features)


def slice_graph_batch(graph, node_data, edge_data, start, stop):
    """Creates a slice from a batch of graphs.

    Parameters
    ----------
    graph: a `GraphInfo` object describing the batch of graphs.
    node_data: a `torch.Tensor` describing data associated with the nodes.
    edge_data: a `torch.Tensor` describing data associated with the edges.
    start: start index of the slice (inclusive).
    stop: stop index of the slice (exclusive).

    Returns
    -------
    GraphInfo
        graph object representing the given slice
    torch.Tensor
        Node data for the given slice
    torch.Tensor
        Edge data for the given slice
    """
    node_start = graph.node_offsets[start]
    edge_start = graph.edge_offsets[start]

    node_stop = graph.node_offsets[stop]
    edge_stop = graph.edge_offsets[stop]

    slice_graph = graph[start:stop]

    slice_node_data = node_data[node_start:node_stop] if node_data is not None else None
    slice_edge_data = edge_data[edge_start:edge_stop] if edge_data is not None else None

    return slice_graph, slice_node_data, slice_edge_data


__all__ = ['offsets_from_counts', 'scopes_from_offsets', 'SparseFeatureBatch', 'GraphInfo']
