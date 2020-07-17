"""This module provides utilities and generic build blocks for graph neural networks."""

import contextlib
import torch

def autograd_range(name):
    """ Creates an autograd range for pytorch autograd profiling
    """
    return torch.autograd.profiler.record_function(name)


def aggregate_by_incidence(values: torch.Tensor, incidence: torch.Tensor,
                           transform_edge_messages=None, transform_edge_messages_args=None,
                           output_size=None):
    """Aggregates values according to an incidence matrix.

    Effectively computes the following operation:

    .. code-block:: python

        output[i] = values[incidence[1, incidence[0] == i]].sum(axis=0)

    This operation essentially implements a sparse-matrix multiplication in coo format in a naive way.
    Optimization opportunity: write using actual cuSparse.

    Parameters
    ----------
    values : torch.Tensor
        A tensor of rank at least 2
    incidence : torch.Tensor
        a `[2, k]` tensor
    transform_edge_messages : function, optional
        an arbitrary function which transforms edge messages.
    transform_edge_messages_args : any
        Arbitrary set of arguments that are passed to the `transform_edge_messages` function.
    output_size : List[int], optional
        if not `None`, the size of the output tensor. Otherwise, we assume the output tensor
        is the same size as `values`.

    Returns
    -------
    torch.Tensor
        The output tensor, of the same rank as values.
    """
    if output_size is None:
        output_size = values.shape[0]

    with autograd_range('broadcast_messages'):
        # broadcast node values to edge messages
        edge_messages = values.index_select(0, incidence[1])

    if transform_edge_messages is not None:
        # apply transformation to edge messages if necessary.
        if transform_edge_messages_args is None:
            transform_edge_messages_args = tuple()

        with autograd_range('transform_messages'):
            edge_messages = transform_edge_messages(edge_messages, *transform_edge_messages_args)

    with autograd_range('aggregate_messages'):
        # collect edge messages into node values
        output = values.new_zeros([output_size] + list(edge_messages.shape[1:]))
        output.index_add_(0, incidence[0], edge_messages)

    return output


class MessagePassingNetwork(torch.nn.Module):
    """ Custom configurable message-passing network.

    This class implements the main plumbing for a message passing network.
    but exposes points that can be configured to easily create different variants of the networks.
    """
    def __init__(self, depth, message_aggregation_network, transform_edge_messages=None):
        """ Creates a new module representing the message passing network.
        Parameters
        ----------
        depth : int
            number of message passing iterations to execute.
        message_aggregation_network : torch.nn.Module
            A module representing the model used to compute the embeddings
            to be used at the next step. This model receives the array
            of messages corresponding to the sum of the propagated messages, and the array
            of previous node embeddings.
        transform_edge_messages : torch.nn.Module
            A module representing the model used to transform
            edge messages at each step. See `aggregate_by_incidence`.
        """
        super(MessagePassingNetwork, self).__init__()

        self.depth = depth
        self.message_aggregation_network = message_aggregation_network
        self.transform_edge_messages = transform_edge_messages


    __constants__ = ["depth"]

    def forward(self, node_embedding, incidence, edge_transform_args=None):
        """Forward function for the message passing network.

        Parameters
        ----------
        node_embedding : torch.Tensor
            Tensor of shape `[num_nodes, ...]` representing the data at each node in the graph.
        incidence : torch.Tensor
            tensor of shape `[2, num_edges]` representing edge incidence in the graph
        edge_transform_args : any
            A tuple of further arguments to be passed to the edge transformation network.

        Returns
        -------
        torch.Tensor
            The final node embedding values after the message passing has been carried out.
        """
        # Compute message passing along edges
        with autograd_range("propagate_messages"):
            for _ in range(self.depth):
                activation = aggregate_by_incidence(
                    node_embedding, incidence, self.transform_edge_messages, edge_transform_args)
                node_embedding = self.message_aggregation_network(activation, node_embedding)

        return node_embedding


class ConcatenateLinear(torch.nn.Module):
    """A torch module which concatenates several inputs and mixes them using a linear layer. """
    def __init__(self, left_size, right_size, output_size):
        """Creates a new concatenating linear layer.

        Parameters
        ----------
        left_size : int
            Size of the left input
        right_size : int
            Size of the right input
        output_size : int
            Size of the output.
        """
        super(ConcatenateLinear, self).__init__()

        self.left_size = left_size
        self.right_size = right_size
        self.output_size = output_size

        self._linear = torch.nn.Linear(left_size + right_size, output_size)

    def forward(self, left, right):
        return self._linear(torch.cat((left, right), dim=-1))


class Sequential(torch.nn.Module):
    """ Similar to `torch.nn.Sequential`, except can pass through modules which
    take multiple input arguments, and return tuples.
    """
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self._sequence_modules = torch.nn.ModuleList(args)

    def forward(self, *args):
        for module in self._sequence_modules:
            if not isinstance(args, (list, tuple)):
                args = [args]
            args = module(*args)

        return args
