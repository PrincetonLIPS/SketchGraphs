"""This module implements a basic constraint checker for a solved graph.

This module implements a number of functions to help check basic relational constraints
between entities. Note that only checking is implemented: this is not an implementation
of a solver, and cannot solve for the desired constraints.

"""

import numpy as  np
from numpy.linalg import norm

from ._entity import Point, Line, Circle, Arc, SubnodeType, ENTITY_TYPE_TO_CLASS, Entity
from ._constraint import ConstraintType
from .sequence import NodeOp, EdgeOp


def get_entity_by_idx(seq, idx: int) -> Entity:
    """Returns the entity or sub-entity corresponding to idx.

    Parameters
    ----------
    seq : List[Union[NodeOp, EdgeOp]]
        A list of node and edge operations representing the construction sequence.
    idx : int
        An integer representing the index of the desired entity.

    Returns
    -------
    Entity
        An entity object representing the entity corresponding to the `NodeOp` at the given index.
    """
    node_ops = [op for op in seq if isinstance(op, NodeOp)]
    label = node_ops[idx].label

    def _entity_from_op(op):
        entity = ENTITY_TYPE_TO_CLASS[op.label]('dummy_id')
        for param_id, val in op.parameters.items():
            setattr(entity, param_id, val)
        return entity

    if not isinstance(label, SubnodeType):
        return _entity_from_op(node_ops[idx])

    for i in reversed(range(idx)):
        this_label = node_ops[i].label
        if not isinstance(this_label, SubnodeType):
            parent = _entity_from_op(node_ops[i])
            break

    if label == SubnodeType.SN_Start:
        return parent.start_point
    elif label == SubnodeType.SN_End:
        return parent.end_point
    elif label == SubnodeType.SN_Center:
        return parent.center_point

    raise ValueError('Could not find entity corresponding to idx.')


def check_edge_satisfied(seq, op: EdgeOp):
    """Determines whether the given EdgeOp instance is geometrically satisfied in the graph represented by `seq`.

    Parameters
    ----------
    seq : List[Union[EdgeOp, NodeOp]]
        A construction sequence representing the underlying graph.
    op : EdgeOp
        An edge op to check for validity in the current graph.

    Returns
    -------
    bool
        `True` if the current edge constraint is satisified, otherwise `False`.

    Raises
    ------
    ValueError
        If the current constraint is not supported (e.g. its type is not supported),
        or it refers to an external entity, which is not supported.
    """
    if 0 in op.references:
        raise ValueError('External constraints not supported.')

    entities = [get_entity_by_idx(seq, ref) for ref in op.references]

    try:
        constraint_f = CONSTRAINT_BY_LABEL[op.label]
    except KeyError:
        raise ValueError('%s not currently supported.' % op.label)

    return constraint_f(*entities)


def get_sorted_types(entities):
    """Obtains the types and sorts the entities based on their type order.

    Parameters
    ----------
    entities : iterable of `Entity`
        An list of entities to be sorted.

    Returns
    -------
    types : List
        A list of types representing the type of each entity
    entities : List
        A list of entities, containing the same elements as the input iterable,
        but sorted in the order given by types.
    """
    types = [Point if isinstance(ent, np.ndarray) else type(ent) for ent in entities]
    type_names = [t.__name__ for t in types]
    idxs = np.argsort(type_names)
    return [types[idx] for idx in idxs], [entities[idx] for idx in idxs]


def _ensure_array(point):
    if isinstance(point, np.ndarray):
        return point
    else:
        return np.array([point.x, point.y])


def coincident(*entities):
    types, entities = get_sorted_types(entities)

    if types == [Point, Point]:
        return np.allclose(_ensure_array(entities[0]), _ensure_array(entities[1]))

    elif types == [Line, Point]:
        vec1 = entities[0].end_point - entities[0].start_point
        vec2 = _ensure_array(entities[1]) - entities[0].start_point
        return np.isclose(np.cross(vec1, vec2), 0)

    elif types == [Line, Line]:
        return coincident(entities[0], entities[1].start_point) and coincident(entities[0], entities[1].end_point)

    elif types in [[Arc, Point], [Circle, Point]]:
        circle_or_arc, point = entities
        dist = norm(_ensure_array(point) - circle_or_arc.center_point)
        return np.isclose(circle_or_arc.radius, dist)

    elif types in [[Circle, Circle], [Arc, Arc], [Arc, Circle]]:
        return np.allclose([entities[0].xCenter, entities[0].yCenter, entities[0].radius],
                           [entities[1].xCenter, entities[1].yCenter, entities[1].radius])

    else:
        return None


def parallel(*ents):
    types, ents = get_sorted_types(ents)

    if types == [Line, Line]:
        vec1 = ents[0].end_point - ents[0].start_point
        vec2 = ents[1].end_point - ents[1].start_point
        return np.isclose(np.cross(vec1, vec2), 0)

    else:
        return None


def horizontal(*ents):
    types, ents = get_sorted_types(ents)

    if types == [Line]:
        return horizontal(ents[0].start_point, ents[0].end_point)
    elif types == [Point, Point]:
        return np.isclose(ents[0][1], ents[1][1], atol=1e-6)
    else:
        return None


def vertical(*ents):
    types, ents = get_sorted_types(ents)

    if types == [Line]:
        return vertical(ents[0].start_point, ents[0].end_point)
    elif types == [Point, Point]:
        return np.isclose(ents[0][0], ents[1][0], atol=1e-6)
    else:
        return None


def perpendicular(*ents):
    types, ents = get_sorted_types(ents)

    if types == [Line, Line]:
        vec1 = ents[0].start_point - ents[0].end_point
        vec2 = ents[1].start_point - ents[1].end_point
        return np.isclose(np.dot(vec1, vec2), 0)
    else:
        return None


def tangent(*ents):
    types, ents = get_sorted_types(ents)

    if types == [Circle, Line]:
        circle, line = ents
        p1, p2 = (line.start_point, line.end_point)
        p3 = circle.center_point

        line_dir = p2 - p1
        line_dir_norm = norm(line_dir)

        if np.abs(line_dir_norm) < 1e-6:
            dist = norm(p1 - p3)
        else:
            dist = norm(np.cross(line_dir, p1-p3)) / line_dir_norm

        return np.isclose(circle.radius, dist)

    elif types == [Arc, Line]:
        arc, line = ents
        circle = Circle('', xCenter=arc.xCenter, yCenter=arc.yCenter, radius=arc.radius)
        return tangent(circle, line)
    elif types in [[Arc, Arc], [Arc, Circle], [Circle, Circle]]:
        dist = norm(ents[1].center_point - ents[0].center_point)
        return (np.isclose(dist, ents[0].radius + ents[1].radius)
                or np.isclose(dist, np.abs(ents[0].radius - ents[1].radius)))
    else:
        return None


def equal(*ents):
    types, ents = get_sorted_types(ents)

    if types == [Line, Line]:
        line0, line1 = ents
        vec0 = line0.end_point - line0.start_point
        vec1 = line1.end_point - line1.start_point
        return np.isclose(norm(vec0), norm(vec1))

    elif types in [[Circle, Circle], [Arc, Arc], [Arc, Circle]]:
        return np.isclose(ents[0].radius, ents[1].radius)

    else:
        return None


def midpoint(*ents):
    types, ents = get_sorted_types(ents)

    if types == [Line, Point]:
        line, point = ents
        mid_coords = (line.start_point + line.end_point) / 2
        return np.allclose(mid_coords, _ensure_array(point))

    else:
        return None


def concentric(*ents):
    types, ents = get_sorted_types(ents)

    if types in [[Circle, Circle], [Arc, Arc], [Arc, Circle]]:
        return coincident(ents[0].center_point, ents[1].center_point)

    else:
        return None


CONSTRAINT_BY_LABEL = {
    ConstraintType.Coincident: coincident,
    ConstraintType.Parallel: parallel,
    ConstraintType.Horizontal: horizontal,
    ConstraintType.Vertical: vertical,
    ConstraintType.Perpendicular: perpendicular,
    ConstraintType.Tangent: tangent,
    ConstraintType.Equal: equal,
    ConstraintType.Midpoint: midpoint,
    ConstraintType.Concentric: concentric
}
