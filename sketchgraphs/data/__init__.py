"""This module contains the main data representations for the SketchGraphs dataset.

There are two main projections for the data.

The first is the `sketch` projection, which is close to the underlying FeatureScript JSON format from Onshape.
This projection is favourable for interactions with Onshape's API, and serves as a first step for parsing
the raw data from Onshape.

The second projection in the `sequence` projection, and is more
specialized towards applications in machine learning. It is more streamlined, and requires conversion
to interact with Onshape's API. However, it interacts more naturally with machine learning applications.

"""

from .sketch import *
from .sequence import *

from .sketch import Sketch, EntityType, SubnodeType, Entity, GenericEntity, Point, Line, Circle, Arc, Spline, Ellipse, ENTITY_TYPE_TO_CLASS
from .sketch import render_sketch, render_graph
