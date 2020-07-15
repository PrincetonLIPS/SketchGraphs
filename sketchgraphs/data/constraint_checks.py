"""This module implements a basic constraint checker for a solved graph."""

import numpy as  np
from numpy.linalg import norm

from ._entity import Point, Line, Circle, Arc, EntityType, SubnodeType, ENTITY_TYPE_TO_CLASS, Entity
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
        return parent.startP
    elif label == SubnodeType.SN_End:
        return parent.endP
    elif label == SubnodeType.SN_Center:
        return parent.centerP

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


def get_sorted_types(ents):
    types = [type(ent) for ent in ents]
    type_names = [t.__name__ for t in types]
    idxs = np.argsort(type_names)
    return [types[idx] for idx in idxs], [ents[idx] for idx in idxs]


def coincident(*ents):
    types, ents = get_sorted_types(ents)

    if types == [Point, Point]:
        return np.allclose(ents[0].coords, ents[1].coords)

    elif types == [Line, Point]:
        vec1 = ents[0].endP.coords - ents[0].startP.coords
        vec2 = ents[1].coords - ents[0].startP.coords
        return np.isclose(np.cross(vec1, vec2), 0)

    elif types == [Line, Line]:
        return coincident(ents[0], ents[1].startP) and coincident(ents[0], ents[1].endP)

    elif types in [[Arc, Point], [Circle, Point]]:
        circle_or_arc, point = ents
        dist = norm(point.coords - circle_or_arc.centerP.coords)
        return np.isclose(circle_or_arc.radius, dist)

    elif types in [[Circle, Circle], [Arc, Arc], [Arc, Circle]]:
        return np.allclose([ents[0].xCenter, ents[0].yCenter, ents[0].radius],
                           [ents[1].xCenter, ents[1].yCenter, ents[1].radius])

    else:
        return None


def parallel(*ents):
    types, ents = get_sorted_types(ents)

    if types == [Line, Line]:
        vec1 = ents[0].endP.coords - ents[0].startP.coords
        vec2 = ents[1].endP.coords - ents[1].startP.coords
        return np.isclose(np.cross(vec1, vec2), 0)

    else:
        return None


def horizontal(*ents):
    types, ents = get_sorted_types(ents)

    if types == [Line]:
        return horizontal(ents[0].startP, ents[0].endP)

    elif types == [Point, Point]:
        return np.isclose(ents[0].y, ents[1].y)

    else:
        return None


def vertical(*ents):
    types, ents = get_sorted_types(ents)

    if types == [Line]:
        return vertical(ents[0].startP, ents[0].endP)

    elif types == [Point, Point]:
        return np.isclose(ents[0].x, ents[1].x)

    else:
        return None


def perpendicular(*ents):
    types, ents = get_sorted_types(ents)

    if types == [Line, Line]:
        vec1 = ents[0].endP.coords - ents[0].startP.coords
        vec2 = ents[1].endP.coords - ents[1].startP.coords
        return np.isclose(np.dot(vec1, vec2), 0)

    else:
        return None


def tangent(*ents):
    types, ents = get_sorted_types(ents)

    if types == [Circle, Line]:
        circle, line = ents
        p1, p2 = (line.startP.coords, line.endP.coords)
        p3 = circle.centerP.coords

        line_dir = p2 - p1
        line_dir_norm = norm(line_dir)

        if np.abs(line_dir_norm) < 1e-6:
            dist = 0
        else:
            dist = norm(np.cross(line_dir, p1-p3)) / line_dir_norm

        return np.isclose(circle.radius, dist)

    elif types == [Arc, Line]:
        arc, line = ents
        circle = Circle('', xCenter=arc.xCenter, yCenter=arc.yCenter, radius=arc.radius)
        return tangent(circle, line)

    elif types in [[Arc, Arc], [Arc, Circle], [Circle, Circle]]:
        dist = norm(ents[1].centerP.coords - ents[0].centerP.coords)
        return np.isclose(dist, ents[0].radius + ents[1].radius)

    else:
        return None


def equal(*ents):
    types, ents = get_sorted_types(ents)

    if types == [Line, Line]:
        line0, line1 = ents
        vec0 = line0.endP.coords - line0.startP.coords
        vec1 = line1.endP.coords - line1.startP.coords
        return np.isclose(norm(vec0), norm(vec1))

    elif types in [[Circle, Circle], [Arc, Arc], [Arc, Circle]]:
        return np.isclose(ents[0].radius, ents[1].radius)

    else:
        return None


def midpoint(*ents):
    types, ents = get_sorted_types(ents)

    if types == [Line, Point]:
        line, point = ents
        mid_coords = (line.startP.coords + line.endP.coords) / 2
        return np.allclose(mid_coords, point.coords)

    else:
        return None


def concentric(*ents):
    types, ents = get_sorted_types(ents)

    if types in [[Circle, Circle], [Arc, Arc], [Arc, Circle]]:
        return coincident(ents[0].centerP, ents[1].centerP)

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
