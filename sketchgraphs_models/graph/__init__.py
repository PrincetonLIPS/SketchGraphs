""" This module contains the main implementation for the graph models.

The graph models represent a family of models which recursively build
the sketch by viewing the sketch as a graph of entities and constraints.
As described in the paper, the baseline generative model does not consider
entity (primitive) coordinates and relies on constraints to determine
the final configuration of the solved sketch.

"""
