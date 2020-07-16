"""This module contains the implementation and data for heuristic degrees of freedom computation."""

import numpy as np

from ._entity import EntityType, SubnodeType
from ._constraint import ConstraintType
from .sequence import EdgeOp, NodeOp


NODE_DOF = {
    EntityType.Point: 2,
    EntityType.Line: 4,
    EntityType.Circle: 3,
    EntityType.Arc: 5,
}

EDGE_DOF_REMOVED = {
    ConstraintType.Angle: {
        (EntityType.Line, EntityType.Line): 1},
    ConstraintType.Centerline_Dimension: {
        (EntityType.Arc, EntityType.Line): 0,
        (EntityType.Circle, EntityType.Line): 0,
        (EntityType.Line, EntityType.Line): 0,
        (EntityType.Line, EntityType.Point): 0},
    ConstraintType.Coincident: {
        (EntityType.Arc, EntityType.Arc): 3,
        (EntityType.Arc, EntityType.Circle): 3,
        (EntityType.Arc, EntityType.Point): 1,
        (EntityType.Circle, EntityType.Circle): 3,
        (EntityType.Circle, EntityType.Point): 1,
        (EntityType.Line, EntityType.Line): 2,
        (EntityType.Line, EntityType.Point): 1,
        (EntityType.Point, EntityType.Point): 2},
    ConstraintType.Concentric: {
        (EntityType.Arc, EntityType.Arc): 2,
        (EntityType.Arc, EntityType.Circle): 2,
        (EntityType.Arc, EntityType.Point): 2,
        (EntityType.Circle, EntityType.Circle): 2,
        (EntityType.Circle, EntityType.Point): 2,
        (EntityType.Point, EntityType.Point): 2},
    ConstraintType.Diameter: {
        (EntityType.Arc, EntityType.Arc): 1, 
        (EntityType.Circle, EntityType.Circle):1},
    ConstraintType.Distance: {
        (EntityType.Arc, EntityType.Arc): 1,
        (EntityType.Arc, EntityType.Circle): 1,
        (EntityType.Arc, EntityType.Line): 1,
        (EntityType.Arc, EntityType.Point): 1,
        (EntityType.Circle, EntityType.Circle): 1,
        (EntityType.Circle, EntityType.Line): 1,
        (EntityType.Circle, EntityType.Point): 1,
        (EntityType.Line, EntityType.Line): 1,
        (EntityType.Line, EntityType.Point): 1,
        (EntityType.Point, EntityType.Point): 1},
    ConstraintType.Equal: {
        (EntityType.Arc, EntityType.Arc): 1,
        (EntityType.Arc, EntityType.Circle): 1,
        (EntityType.Circle, EntityType.Circle): 1,
        (EntityType.Line, EntityType.Line): 1},
    ConstraintType.Fix: {
        (EntityType.Arc, EntityType.Arc): 3,
        (EntityType.Circle, EntityType.Circle): 3,
        (EntityType.Line, EntityType.Line): 2,
        (EntityType.Point, EntityType.Point): 2},
    ConstraintType.Horizontal: {
        (EntityType.Line, EntityType.Line): 1,
        (EntityType.Point, EntityType.Point): 1},
    ConstraintType.Intersected: {},
    ConstraintType.Length: {
        (EntityType.Arc, EntityType.Arc): 1,
        (EntityType.Line, EntityType.Line): 1},
    ConstraintType.Midpoint: {
        (EntityType.Arc, EntityType.Point): 2,
        (EntityType.Line, EntityType.Point): 2},
    ConstraintType.Normal: {
        (EntityType.Arc, EntityType.Line): 1,
        (EntityType.Circle, EntityType.Line): 1},
    ConstraintType.Offset: {
        (EntityType.Arc, EntityType.Arc): 2,
        (EntityType.Arc, EntityType.Circle): 2,
        (EntityType.Circle, EntityType.Circle): 2,
        (EntityType.Line, EntityType.Line): 1},
    ConstraintType.Parallel: {
        (EntityType.Line, EntityType.Line): 1},
    ConstraintType.Perpendicular: {
        (EntityType.Line, EntityType.Line): 1},
    ConstraintType.Radius: {
        (EntityType.Arc, EntityType.Arc): 1,
        (EntityType.Circle, EntityType.Circle): 1},
    ConstraintType.Subnode: {
        (EntityType.Arc, EntityType.Point): 0,
        (EntityType.Circle, EntityType.Point): 0,
        (EntityType.Line, EntityType.Point): 0},
    ConstraintType.Tangent: {
        (EntityType.Arc, EntityType.Arc): 1,
        (EntityType.Arc, EntityType.Circle): 1,
        (EntityType.Arc, EntityType.Line): 1,
        (EntityType.Circle, EntityType.Circle): 1,
        (EntityType.Circle, EntityType.Line): 1},
    ConstraintType.Vertical: {
        (EntityType.Line, EntityType.Line): 1,
        (EntityType.Point, EntityType.Point): 1}
}


def get_node_label_for_dof(label) -> EntityType:
    """Get the node label to be used for degrees of freedom computation.

    When computing DOF, subnode labels are merged to point as they represent points.

    Parameters
    ----------
    label : Union[EntityType, SubnodeType]
        The original node label

    Returns
    -------
    EntityType
        The corresponding `EntityType` to be used for DOF computation.
    """
    return EntityType.Point if isinstance(label, SubnodeType) else label


def _get_dof_removed_for_edge(edge_op: EdgeOp, nodes):
    ref_types = [get_node_label_for_dof(nodes[r].label) for r in edge_op.references]

    if len(ref_types) == 1:
        ref_types = ref_types + ref_types
    elif len(ref_types) > 2:
        return 0
    t1, t2 = ref_types

    if EntityType.External in ref_types:
        return 0


    dof_dict = EDGE_DOF_REMOVED.get(edge_op.label)
    if dof_dict is None:
        return 0

    if (t1, t2) in dof_dict:
        return dof_dict[(t1, t2)]
    if (t2, t1) in dof_dict:
        return dof_dict[(t2, t1)]

    return 0


def get_sequence_dof(seq):
    """Returns array of total DoF contribution from each op.

    Parameters
    ----------
    seq : Iterable of NodeOp or EdgeOp
        The construction sequence of the sketch to analyze

    Returns
    -------
    np.ndarray
        An integer array representing the number of degrees of freedom lost or gained
        at each construction step.
    """
    out = np.zeros(len(seq), dtype=np.int32)
    nodes = []
    for i, op in enumerate(seq):
        if isinstance(op, NodeOp):
            nodes.append(op)
            out[i] = NODE_DOF.get(op.label, 0)
        elif isinstance(op, EdgeOp):
            out[i] = -1*_get_dof_removed_for_edge(op, nodes)
    return out


__all__ = ['get_sequence_dof', 'get_node_label_for_dof']
