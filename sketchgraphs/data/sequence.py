"""This module contains the implementation for the "sequence" projection of SketchGraphs.

The sequence projection (consisting of `NodeOp` representing entities, and `EdgeOp` representing constraints),
is a streamlined representation of sketches, adapted for machine learning applications.

"""

import collections
import typing
import json

from ._entity import Entity, EntityType, SubnodeType, ENTITY_TYPE_TO_CLASS
from ._constraint import Constraint, ConstraintType, ConstraintParameterType, LocalReferenceParameter
from ._constraint import NUMERIC_IDS, ENUM_IDS, BOOL_IDS, QuantityParameter, EnumParameter, BooleanParameter, BooleanValue
from .sketch import Sketch


class NodeOp(typing.NamedTuple):
    """This class represents a node (or entity) operation in a construction sequence.

    An entity is specified by a label, along with a dictionary of parameters.
    """
    label: EntityType
    parameters: dict = {}


class EdgeOp(typing.NamedTuple):
    """This class represents an edge (or constraint) operation in a construction sequence.

    An edge is specified by a label, a variable number of targets, and a dictionary of parameters.
    Targets for an edge are nodes, which are referred to using their index in the construction sequence.
    """
    label: ConstraintType
    references: typing.Tuple[int, ...]
    parameters: dict = {}


def _get_entity_parameters(entity: Entity):
    parameter_ids = type(entity).bool_ids + type(entity).float_ids
    return {param_id: getattr(entity, param_id) for param_id in parameter_ids}


def _get_constraint_parameters(constraint: Constraint):
    parameters = {}
    if not constraint.type.has_parameters:
        return parameters

    schema = [param.parameterId for param in constraint.parameters]
    ref_schema = constraint.type.normalize(schema)

    if not ref_schema:
        return parameters  # return empty parameters for unsupported schema

    for param in constraint.parameters:
        param_id = param.parameterId
        if param_id in ref_schema:
            if param.type == ConstraintParameterType.Quantity:
                parameters[param_id] = param.expression
            elif param.type in (ConstraintParameterType.Enum, ConstraintParameterType.Boolean):
                parameters[param_id] = param.value

    return parameters


def sketch_to_sequence(sketch: Sketch, ignore_invalid_constraints=True) -> typing.List[typing.Union[NodeOp, EdgeOp]]:
    """Creates a sequence representation in terms of `NodeOp` and `EdgeOp` from the given sketch.

    All the entities in the sketch are converted, along with the respective constraints.
    Constraints are expressed in terms of entity indices (instead of original string identifiers).
    The sequence is ordered such that the nodes are in original insertion order, and the edges
    are placed such that they directly follow the last node they reference.

    Parameters
    ----------
    sketch : Sketch
        The sketch object to convert to sequence.
    ignore_invalid_constraints : bool
        If True, indicates that invalid constraints should be ignored. Otherwise, raises a ValueError
        on encountering an invalid constraint.

    Returns
    -------
    list
        A list of construction operations

    Raises
    ------
    ValueError
        If ``ignore_invalid_constraints`` is set to False, a ValueError will be raised
        upon encountering invalid constraints (e.g. constraints which refer to non-existing nodes,
        or whose parameters cannot be processed according to the specified schema).
    """
    node_sequence = []
    edge_sequence = []
    node_idx_map = {}

    node_sequence.append(NodeOp(EntityType.External))
    node_idx_map['External'] = 0

    for node_id, node in sketch.entities.items():
        if node.type == EntityType.Unknown:
            continue

        current_node_index = len(node_sequence)
        node_idx_map[node_id] = current_node_index
        node_sequence.append(NodeOp(node.type, _get_entity_parameters(node)))

        for subnode_type, subnode_id in zip(node.get_subnode_types(), node.get_subnode_ids()):
            current_subnode_index = len(node_sequence)
            node_idx_map[subnode_id] = current_subnode_index
            node_sequence.append(NodeOp(subnode_type))
            edge_sequence.append(EdgeOp(ConstraintType.Subnode, (current_subnode_index, current_node_index)))

    node_sequence.append(NodeOp(EntityType.Stop))

    for edge in sketch.constraints.values():
        if edge.type == ConstraintType.Unknown:
            continue

        references = []
        ignore_reference = False
        for r in edge.get_references():
            r_idx = node_idx_map.get(r)

            if r_idx is not None:
                references.append(r_idx)
                continue

            if ignore_invalid_constraints:
                ignore_reference = True
                break
            else:
                raise ValueError('Invalid constraint reference {0}'.format(r))

        if ignore_reference:
            continue

        references.sort(reverse=True)
        references = tuple(references)
        parameters = _get_constraint_parameters(edge)

        if edge.type.has_parameters and not parameters:
            if ignore_invalid_constraints:
                # The constraint is expected to have parameters but they could not be obtained.
                # If set to ignore, skip including this constraint in the sequence
                continue
            raise ValueError('Invalid constraint: parameters could not be processed.')

        edge_sequence.append(EdgeOp(edge.type, references, parameters))

    edge_sequence.sort(key=lambda x: x.references[0])

    sequence = []

    j = 0

    for i, node in enumerate(node_sequence):
        sequence.append(node)

        while j < len(edge_sequence) and edge_sequence[j].references[0] == i:
            sequence.append(edge_sequence[j])
            j += 1

    # The above should exhaust all edges as no edges refer to the last (stop) node.
    assert j == len(edge_sequence)

    return sequence


def sketch_from_sequence(seq) -> Sketch:
    """Builds a Sketch object from the given sequence.

    Parameters
    ----------
    seq : List[Union[NodeOp, EdgeOp]]
        A construction sequence representing a sketch.

    Returns
    -------
    Sketch
        A sketch object representing the given construction sequence.
    """
    subnode_label_to_string = {
        SubnodeType.SN_Start: 'start',
        SubnodeType.SN_End: 'end',
        SubnodeType.SN_Center: 'center'
    }

    entities = collections.OrderedDict()
    constraints = collections.OrderedDict()

    main_node_map = {}
    node_idx = -1
    last_main_idx = node_idx
    constraint_idx = 1

    def create_entity_id(node_idx):
        """Helper function for creating entity id - manages sub-entities"""
        if node_idx not in main_node_map:
            return str(node_idx)
        else:
            main_idx, subnode_label = main_node_map[node_idx]
            return '%i.%s' % (main_idx, subnode_label_to_string[subnode_label])

    for op in seq:
        if isinstance(op, NodeOp):
            node_idx += 1
            # Check if this is a main (non-sub) node
            if isinstance(op.label, EntityType):
                last_main_idx = node_idx
                entityId = str(node_idx)
                # Don't create External or Stop entity
                if op.label not in (EntityType.External, EntityType.Stop):
                    entities[entityId] = ENTITY_TYPE_TO_CLASS[op.label](entityId)
                    # Extract parameters for numerical entities
                    for param_id, val in op.parameters.items():
                        setattr(entities[entityId], param_id, val)
            else:
                # Mark main reference of this subnode
                main_node_map[node_idx] = (last_main_idx, op.label)

        else:
            if 0 in op.references:
                # Skipping external constraints for now
                continue
            if op.label == ConstraintType.Subnode:
                # Don't include subnode edges in constraints
                continue

            constraint_ent_id = 'c_%i' % constraint_idx
            constraint_idx += 1
            constraint_type = op.label

            # Adjust reference parameter ids if necessary
            if constraint_type == ConstraintType.Midpoint:
                param_ids = ['local0', 'local1']
            else:
                param_ids = ['localFirst', 'localSecond']

            param1 = LocalReferenceParameter(param_ids[0], create_entity_id(op.references[-1]))
            params = [param1]

            if len(op.references) > 1:
                # Non self-constraint
                param2 = LocalReferenceParameter(param_ids[1], create_entity_id(op.references[0]))
                params.append(param2)

            # Extract parameters for numerical constraints
            params_are_ok = True
            for param_id, val in op.parameters.items():
                if param_id in NUMERIC_IDS:
                    if val == 'OTHER':  # skip unsupported values
                        params_are_ok = False
                    val = val.replace('METER', 'm')
                    val = val.replace('DEGREE', 'deg')
                    params.append(QuantityParameter(param_id, '0.0', val))  # val is really expression here
                elif param_id in ENUM_IDS:
                    params.append(EnumParameter(param_id, val))
                elif param_id in BOOL_IDS:
                    val = bool(BooleanValue[val])
                    params.append(BooleanParameter(param_id, val))
                else:
                    raise ValueError("Unknown parameter id.")

            if params_are_ok:
                constraints[constraint_ent_id] = Constraint(constraint_ent_id, constraint_type, params)

    return Sketch(entities=entities, constraints=constraints)


def pgvgraph_from_sequence(seq):
    """Builds a pgv.AGraph from the sequence.

    Hyperedges are not supported in the resulting graph.

    Parameters
    ----------
    seq : List[Union[NodeOp, EdgeOp]]
        A construction sequence representing a sketch.

    Returns
    -------
    pgv.AGraph
        A pygraphviz AGraph corresponding to the given sequence.
    """
    import pygraphviz as pgv

    graph = pgv.AGraph(strict=False)
    idx = 0
    for op in seq:
        if isinstance(op, NodeOp):
            if op.label != EntityType.Stop:
                graph.add_node(idx, label=op.label.name, parameters=json.dumps(op.parameters), index=idx)
                idx += 1
    idx = 0
    for op in seq:
        if isinstance(op, EdgeOp):
            if len(op.references) > 2:
                continue  # hyperedges not supported
            node_a = op.references[0]
            if len(op.references) == 1:
                node_b = node_a
            elif len(op.references) == 2:
                node_b = op.references[1]
            else:
                raise ValueError("Invalid number of references in EdgeOp.")
            graph.add_edge(node_a, node_b, label=op.label.name, parameters=json.dumps(op.parameters), index=idx)
            idx += 1
    return graph