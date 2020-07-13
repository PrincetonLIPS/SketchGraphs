"""This module implements the basic definitions for targets."""

import enum

from sketchgraphs.data import sketch as datalib, sequence as data_sequence

NODE_TYPES_PREDICTED = list(datalib.EntityType)
EDGE_TYPES_PREDICTED = list(x for x in datalib.ConstraintType if x != datalib.ConstraintType.Subnode)
NODE_TYPES = list(datalib.EntityType) + list(datalib.SubnodeType)
EDGE_TYPES = list(datalib.ConstraintType)


EDGE_IDX_MAP = {t: i for i, t in enumerate(datalib.ConstraintType)}
NODE_IDX_MAP = {t: i for i, t in enumerate(list(datalib.EntityType) + list(datalib.SubnodeType))}
NODE_IDX_MAP_REVERSE = {i: t for i, t in enumerate(list(datalib.EntityType) + list(datalib.SubnodeType))}
EDGE_IDX_MAP_REVERSE = {i: t for i, t in enumerate(datalib.ConstraintType)}


class TargetType(enum.IntEnum):
    EdgeCategorical = 0
    EdgeAngle = 1
    EdgeLength = 2
    EdgeDistance = 3
    EdgeDiameter = 4
    EdgeRadius = 5
    NodeGeneric = 6
    NodeArc = 7
    NodeCircle = 8
    NodeLine = 9
    NodePoint = 10
    Subnode = 11

    @staticmethod
    def from_op(op):
        return TargetType.from_label(op.label)

    @staticmethod
    def from_label(label):
        if isinstance(label, datalib.SubnodeType):
            return TargetType.Subnode
        elif isinstance(label, datalib.EntityType):
            return _entity_to_target_type.get(label, TargetType.NodeGeneric)
        elif isinstance(label, datalib.ConstraintType):
            return _edge_to_target_type.get(label, TargetType.EdgeCategorical)
        else:
            raise ValueError('label must be one of SubnodeType, EntityType or ConstraintType.')

    @staticmethod
    def from_edge_label_int(x):
        fake_edge_op = datalib.EdgeOp(x, 0, 0)
        return TargetType.from_op(fake_edge_op)

    @staticmethod
    def from_node_label_int(x):
        fake_node_op = datalib.NodeOp(x)
        return TargetType.from_op(fake_node_op)

    @staticmethod
    def edge_types():
        return (TargetType.EdgeCategorical,) + TargetType.numerical_edge_types()

    @staticmethod
    def numerical_edge_types():
        return (
            TargetType.EdgeAngle,
            TargetType.EdgeLength,
            TargetType.EdgeDistance,
            TargetType.EdgeDiameter,
            TargetType.EdgeRadius
        )

    @staticmethod
    def numerical_node_types():
        return (
            TargetType.NodeArc,
            TargetType.NodeCircle,
            TargetType.NodeLine,
            TargetType.NodePoint
        )

    @staticmethod
    def node_types():
        return (TargetType.NodeGeneric,) + TargetType.numerical_node_types()

_str_to_numerical_edge = {
    t.name.upper(): t
    for t in TargetType.numerical_edge_types()
}

_str_to_numerical_entity = {
    t.name.title(): t
    for t in TargetType.numerical_node_types()
}

_edge_to_target_type = {
    s: TargetType['Edge' + s.name] for s in
    (datalib.ConstraintType.Angle, datalib.ConstraintType.Length, datalib.ConstraintType.Distance,
     datalib.ConstraintType.Diameter, datalib.ConstraintType.Radius)
}

_entity_to_target_type = {
    s: TargetType['Node' + s.name] for s in
    (datalib.EntityType.Arc, datalib.EntityType.Circle, datalib.EntityType.Line, datalib.EntityType.Point)
}
