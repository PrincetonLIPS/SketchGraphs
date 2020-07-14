""" This module contains the components of the graph model that are concerned with handling
numerical features associated with edges and nodes.

"""

import torch

from sketchgraphs_models import nn as sg_nn


class NumericalFeatureEncoding(torch.nn.Module):
    """Encode an array of numerical features.

    This encodes a sequence of features (presented as a sequence of integers)
    into a sequence of vector through an embedding.
    """
    def __init__(self, feature_dims, embedding_dim):
        super(NumericalFeatureEncoding, self).__init__()
        self.feature_dims = list(feature_dims)
        self.register_buffer(
            'feature_offsets',
            torch.cumsum(torch.tensor([0] + self.feature_dims[:-1], dtype=torch.int64), dim=0))
        self.embeddings = torch.nn.Embedding(
            sum(feature_dims), embedding_dim, sparse=False)

    def forward(self, features):
        return self.embeddings(features + self.feature_offsets)


class NumericalFeaturesEmbedding(torch.nn.Module):
    """Transform a sequence of numerical feature vectors into a single vector.

    Currently, this module simply aggregates the features by averaging, although more
    elaborate aggregation schemes (e.g. RNN) could be chosen.
    """
    def __init__(self, embedding_dim):
        super(NumericalFeaturesEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, embeddings):
        return embeddings.mean(axis=-2)


class NumericalFeatureDecoder(torch.nn.Module):
    """Module for decoding numerical feature embeddings to logits.

    Takes an array of features, and decodes them into a a sequence of varying length logits.
    """
    def __init__(self, feature_dims, embedding_dim):
        super(NumericalFeatureDecoder, self).__init__()
        self.feature_dims = feature_dims
        self.linear_logistic = torch.nn.ModuleList([
            torch.nn.Linear(embedding_dim, fd) for fd in feature_dims
        ])

    def forward(self, embeddings):
        logits = []

        for i, linear in enumerate(self.linear_logistic):
            logits.append(linear(embeddings[i]))

        return torch.cat(logits, dim=-1)


class NumericalFeatureReadout(torch.nn.Module):
    """Module for numerical feature readout.

    This module is responsible for producing numerical edge features from the edge label
    and computed embeddings.
    """
    def __init__(self, initial_input, feature_encoders, feature_decoders, sequence_model):
        """Creates a new edge feature readout model.

        Parameters
        ----------
        initial_input : torch.nn.Module
            A module which creates the embedding for the first input slot based on the passed in data.
            It is called with all remaining arguments to the forward method.

        feature_encoders : torch.nn.Module
            A module which encodes the features from the edge for feeding in to network.
            It is called with an integer tensor of size `[batch, num_features]`.

        feature_decoders : torch.nn.Module
            A module which decodes sequence embeddings into an array of logits.

        sequence_model : torch.nn.Module
            The main computational module for this instance, transforms a sequence
            of embedding into another sequence of embedding.
        """
        super(NumericalFeatureReadout, self).__init__()
        self.encoders = feature_encoders
        self.decoders = feature_decoders
        self.sequence_model = sequence_model
        self.initial_input = initial_input


    def forward(self, input_features, *args):
        initial_input = self.initial_input(*args)
        input_embeddings = self.encoders(input_features).permute(1, 0, 2)

        input_sequence = torch.cat((initial_input.expand(1, -1, -1), input_embeddings), dim=0)

        output_sequence, _ = self.sequence_model(input_sequence)

        return self.decoders(output_sequence[:-1])


def edge_decoder_initial_input(embedding_size):
    """Initial input function for edge readouts."""
    return sg_nn.Sequential(
        sg_nn.ConcatenateLinear(embedding_size, embedding_size, embedding_size),
        torch.nn.ReLU(),
        torch.nn.Linear(embedding_size, embedding_size))


def entity_decoder_initial_input(embedding_size):
    """Initial input function for entity readouts."""
    return torch.nn.Identity(embedding_size=embedding_size)


def make_embedding_and_readout(embedding_size: int, feature_dimensions, initial_input_factory):
    """Creates feature embedding and readout networks for the given features.

    Parameters
    ----------
    embedding_size : int
        Dimension of the embeddings to use.
    feature_dimensions : dict
        Dictionary whose values are lists of integers corresponding to the number of outcomes
        for each feature.
    initial_input_factory : int -> torch.nn.Module
        A function which returns a module responsible for transforming the inputs of the readout
        into initial embeddings for the internal sequence model.

    Returns
    -------
    feature_embeddings : dict
        A dictionary containing the feature embedding modules.
    feature_readouts : dict
        A dictionary containing the feature readout modules.
    """

    feature_encodings = {
        k: NumericalFeatureEncoding(dimensions.values(), embedding_size)
        for k, dimensions in feature_dimensions.items()
    }

    feature_embeddings = {
        k.name: torch.nn.Sequential(
            encoding, NumericalFeaturesEmbedding(embedding_size)
        )
        for k, encoding in feature_encodings.items()
    }

    feature_readouts = {
        k.name: NumericalFeatureReadout(
            initial_input=initial_input_factory(embedding_size),
            feature_encoders=feature_encodings[k],
            feature_decoders=NumericalFeatureDecoder(dimensions.values(), embedding_size),
            sequence_model=torch.nn.GRU(embedding_size, embedding_size))
        for k, dimensions in feature_dimensions.items()
    }

    return feature_embeddings, feature_readouts
