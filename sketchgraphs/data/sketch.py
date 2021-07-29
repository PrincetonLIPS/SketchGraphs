"""This module implements parsing and representation for sketches.
"""

from collections import OrderedDict
from typing import Dict

# pylint: disable=invalid-name, too-many-arguments, too-many-return-statements, too-many-instance-attributes, wildcard-import, unused-wildcard-import


from . import _entity
from . import _constraint
from . import _plotting

from ._entity import EntityType, SubnodeType, Entity, GenericEntity, Point, Line, Circle, Arc, Spline, Ellipse, ENTITY_TYPE_TO_CLASS

from ._constraint import *
from ._plotting import render_sketch, render_graph


class Sketch:
    """This class encapsulates a sketch instance.

    A sketch is defined by a list of entities, and a list of constraints between these entities.
    The Sketch class is designed to represent the sketches as obtained from Onshape in a structured
    and faithful manner. In particular, it can round-trip the relevant parts of the JSON representation
    of Onshape's feature-script.
    """
    entities: Dict[str, Entity]
    constraints: Dict[str, Constraint]

    def __init__(self, entities=None, constraints=None):
        if entities is None:
            entities = OrderedDict()
        if constraints is None:
            constraints = OrderedDict()

        self.entities = entities
        self.constraints = constraints

    def to_dict(self) -> dict:
        """Create a dictionary representing this sketch.

        The created dictionary should be compatible with the json represention of the sketch.
        """
        return {
            'entities': [e.to_dict() for e in self.entities.values()],
            'constraints': [c.to_dict() for c in self.constraints.values()]
        }

    @staticmethod
    def from_fs_json(sketch_dict, include_external_constraints=True):
        """Parse primitives and constraints.

        Parameters
        ----------
        include_external_constraints : bool, optional
            If True, indicates that constraints referencing the first external node (representing)
            the origin should be included, otherwise, exclude those from the graph.
        """
        entities = (Entity.from_dict(ed) for ed in sketch_dict['entities'])
        entities_dict = OrderedDict((e.entityId, e) for e in entities)

        constraints = (Constraint.from_dict(cd) for cd in sketch_dict['constraints'])

        if not include_external_constraints:
            constraints = (
                c for c in constraints
                if not any(isinstance(p, ExternalReferenceParameter) for p in c.parameters))

        constraints_dict = OrderedDict((c.identifier, c) for c in constraints)
        return Sketch(entities_dict, constraints_dict)

    @staticmethod
    def from_info(sketch_info):
        """Parse entities given result of `sketch information` call."""
        subnode_suffixes = ('.start', '.end', '.center')  # TODO: dry-ify this
        entities = [Entity.from_info(ed) for ed in sketch_info if not ed['id'].endswith(subnode_suffixes)]
        entities_dict = OrderedDict((e.entityId, e) for e in entities)
        return Sketch(entities=entities_dict)

    def __repr__(self):
        return 'Sketch(n_entities={0}, n_constraints={1})'.format(len(self.entities), len(self.constraints))


__all__ = [
    'Sketch', 'EntityType', 'SubnodeType', 'Entity', 'GenericEntity', 'Point', 'Line',
    'Circle', 'Arc', 'Spline', 'Ellipse', 'ENTITY_TYPE_TO_CLASS'] + _constraint.__all__ + _plotting.__all__
