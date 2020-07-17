""" This module contains the main implementation for the graph models.

The graph models represent a family of models which recursively build
the sketch by viewing the sketch as a graph of entities and constraints.
This model is configurable in the amount of continuous parameters it models.
In particular, by using the `--disable_entity_features` or `--disable_edge_features`
switches, one can disable the modelling of parameters attached to either edges
or entities (or both). Modelling sketches without entity features can lead to
meaningless sketches before solving, but the presence of constraints can often
lead to reasonable solved states.

"""
