"""This module implements parsing and representation of sketch constraints.
"""

import abc
import enum
import typing
import sys

# pylint: disable=invalid-name, too-many-arguments, too-many-return-statements, too-many-instance-attributes


class ConstraintType(enum.IntEnum):
    """This enumeration represents the type of the constraint."""
    Coincident = 0
    Projected = 1
    Mirror = 2
    Distance = 3
    Horizontal = 4
    Parallel = 5
    Vertical = 6
    Tangent = 7
    Length = 8
    Perpendicular = 9
    Midpoint = 10
    Equal = 11
    Diameter = 12
    Offset = 13
    Radius = 14
    Concentric = 15
    Fix = 16
    Angle = 17
    Circular_Pattern = 18
    Pierce = 19
    Linear_Pattern = 20
    Centerline_Dimension = 21
    Intersected = 22
    Silhoutted = 23
    Quadrant = 24
    Normal = 25
    Minor_Diameter = 26
    Major_Diameter = 27
    Rho = 28
    Unknown = 29
    Subnode = 101

    @property
    def _numeric_schema(self):
        """Supported numerical constraint schemas; will be added to over time.

        For Offset we do 2nd most popular since we can only handle 2 entity references atm.
        """
        return {
            ConstraintType.Angle: ('aligned', 'angle', 'clockwise', 'local0',
                                   'local1'),
            ConstraintType.Distance: ('direction', 'halfSpace0', 'halfSpace1',
                                      'length', 'local0', 'local1'),
            ConstraintType.Length: ('direction', 'length', 'local0'),
            ConstraintType.Offset: ('local0', 'local1'),
            ConstraintType.Diameter: ('length', 'local0'),
            ConstraintType.Radius: ('length', 'local0')
        }.get(self)

    @property
    def has_parameters(self) -> bool:
        """A boolean value indicating whether the given constraint has parameters."""
        return self._numeric_schema is not None

    def normalize(self, schema):
        """Normalizes supported schemas for numerical constraintType.

        We will consider identical any schemas that only differ by local# vs. localWord.

        Parameters
        ----------
        schema: an interable containing parameter ID strings
        constraitType: an instance of ConstraintType

        Returns
        -------
        If schema is supported, returns ref_schema, the set of relevant params; otherwise returns False.
        """
        ignored_params = ('labelAngle', 'labelDistance', 'labelRatio', 'radiusDisplay')
        ref_schema = self._numeric_schema
        if ref_schema is None:
            raise ValueError("Only call schema_comp on numerical constraints.")
        norm_schema = []
        for param_id in schema:
            if param_id in ignored_params:
                continue
            # TODO: address these magic strings
            if param_id == 'localFirst':
                param_id = 'local0'
            elif param_id == 'localSecond':
                param_id = 'local1'
            norm_schema.append(param_id)
        if set(norm_schema) == set(ref_schema):
            return set(ref_schema)
        else:
            return False


string_to_constraint_type = {e.name.upper(): e for e in ConstraintType}


NUMERIC_IDS = ('angle', 'length')
ENUM_IDS = ('direction', 'halfSpace0', 'halfSpace1', 'alignment')
BOOL_IDS = ('aligned', 'clockwise')


class DirectionValue(enum.IntEnum):
    """This enumeration represents the options for a direction parameter."""
    MINIMUM = 0
    HORIZONTAL = 1
    VERTICAL = 2


class HalfSpaceValue(enum.IntEnum):
    """This enumeration represents the options for a halfSpace* parameter."""
    LEFT = 0
    RIGHT = 1

class AlignmentValue(enum.IntEnum):
    """This enumeration represents the options for an alignment parameter."""
    ANTI_ALIGNED = 0
    ALIGNED = 1

BooleanValue = enum.IntEnum(
    'BooleanValue',
    [('FALSE', 0), ('TRUE', 1), ('False', 0), ('True', 1)]
)

BooleanValue.__doc__ = "This enumeration represents the options for a boolean parameter."

# sphinx doesn't deal well with the hack we are using here, so only execute
# when sphinx is not found in modules.
if 'sphinx' not in sys.modules:
    BooleanValue._member_map_[False] = BooleanValue.FALSE
    BooleanValue._member_map_[True] = BooleanValue.TRUE


param_id_to_options = {
    'direction': DirectionValue,
    'halfSpace0': HalfSpaceValue,
    'halfSpace1': HalfSpaceValue,
    'alignment': AlignmentValue
}


class ConstraintParameterType(enum.Enum):
    """This enumeration represents the type of the parameter."""
    LocalReference = 0
    ExternalReference = 1
    Quantity = 2
    Enum = 3
    Boolean = 4
    Unknown = 5


def _inspect_parameter_type(param_dict):

    if param_dict['typeName'] == 'BTMParameterQuantity':
        return ConstraintParameterType.Quantity

    if param_dict['typeName'] == 'BTMParameterEnum':
        return ConstraintParameterType.Enum

    if param_dict['typeName'] == 'BTMParameterBoolean':
        return ConstraintParameterType.Boolean

    if param_dict['typeName'] == 'BTMParameterString':
        if 'local' in param_dict['message']['parameterId']:
            return ConstraintParameterType.LocalReference
        if 'external' in param_dict['message']['parameterId']:
            return ConstraintParameterType.ExternalReference

    if param_dict['typeName'] == 'BTMParameterQueryList':
        if 'external' in param_dict['message']['parameterId']:
            return ConstraintParameterType.ExternalReference

    return ConstraintParameterType.Unknown


def _get_parameter_common_attributes(param_dict):
    return (param_dict['message']['parameterId'],)


class ConstraintParameter(abc.ABC):
    """This class represents a parameter for a given constraint.

    Parameters are given by name-value pairs.
    However, note that not all parameter types populate the value field.
    """
    parameterId: str

    def __init__(self, parameterId):
        self.parameterId = parameterId

    @property
    @abc.abstractmethod
    def type(self) -> ConstraintParameterType:
        pass

    @abc.abstractmethod
    def to_dict(self) -> dict:
        """Serializes this given parameter to a dictionary which represents the parameter."""

    @staticmethod
    def from_dict(param_dict):
        """Parses a representation for this parameter from a json dictionary."""
        parameter_type = _inspect_parameter_type(param_dict)
        parameterClass = _parameter_type_to_class.get(parameter_type, GenericParameter)
        return parameterClass.from_dict(param_dict)


class GenericParameter(ConstraintParameter):
    """Generic constraint parameter. """
    data: dict

    def __init__(self, data):
        super(GenericParameter, self).__init__(*_get_parameter_common_attributes(data))
        self.data = data

    @property
    def type(self):
        return ConstraintParameterType.Unknown

    def __repr__(self):
        return f"GenericParam({self.parameterId})"

    def to_dict(self):
        return self.data

    @staticmethod
    def from_dict(param_dict):
        return GenericParameter(param_dict)


class LocalReferenceParameter(ConstraintParameter):
    """This parameter represents a reference to an entity defined in the sketch.

    This class also maintains a normalized version of the entity referenced in the field
    `referenceMain`. That field captures the entityId of the main entity being referenced,
    whereas the value field may contain a reference to some subpart of that entity.
    """
    value: str

    def __init__(self, parameterId, value):
        super(LocalReferenceParameter, self).__init__(parameterId)
        self.value = value
        self.referenceMain = self.get_referenceMain()

    @property
    def type(self):
        return ConstraintParameterType.LocalReference

    def get_referenceMain(self):
        split_value = self.value.split('.')
        if split_value[-1] in ['start', 'end', 'center']:
            return '.'.join(split_value[:-1])
        else:
            return self.value

    def __repr__(self):
        return f"LocalRefParam({self.parameterId}: {self.value})"

    def to_dict(self):
        return {
            'message': {
                'parameterId': self.parameterId,
                'value': self.value,
            },
            'type': 149,
            'typeName': 'BTMParameterString'
        }

    @staticmethod
    def from_dict(param_dict):
        return LocalReferenceParameter(
            *_get_parameter_common_attributes(param_dict), param_dict['message']['value'])


class ExternalReferenceParameter(ConstraintParameter):
    """This class represents an external reference. """
    def __init__(self, parameterId):
        super(ExternalReferenceParameter, self).__init__(parameterId)

    @property
    def type(self):
        return ConstraintParameterType.ExternalReference

    def __repr__(self):
        return f"ExternalRefParam({self.parameterId})"

    def to_dict(self):
        return {
            'message': {
                'parameterId': self.parameterId
            },
            'type': 149,
            'typeName': 'BTMParameterString',
        }

    @staticmethod
    def from_dict(param_dict):
        return ExternalReferenceParameter(*_get_parameter_common_attributes(param_dict))


class QuantityParameter(ConstraintParameter):
    """Quantitative parameter.

    This parameter represents a quantity given by an expression.
    Unfortunately, although there seems to be fields for structured data
    (e.g. `units` and `value`), they are left empty, and only a plain-text
    expression field is provided.
    """
    def __init__(self, parameterId, value, expression):
        super(QuantityParameter, self).__init__(parameterId)
        self.expression = expression
        self.value = value

    @property
    def type(self):
        return ConstraintParameterType.Quantity

    def __repr__(self):
        return f"QuantityParam({self.parameterId}: {self.expression})"

    def to_dict(self):
        return {
            'message': {
                'expression': self.expression,
                'parameterId': self.parameterId,
                'value': self.value
            },
            'type': 147,
            'typeName': 'BTMParameterQuantity'
        }

    @staticmethod
    def from_dict(param_dict):
        msg_dict = param_dict['message']
        return QuantityParameter(
            *_get_parameter_common_attributes(param_dict),
            msg_dict['value'],
            msg_dict['expression'])


class EnumParameter(ConstraintParameter):
    """Categorical parameter."""
    def __init__(self, parameterId, value):
        super(EnumParameter, self).__init__(parameterId)
        self.value = value

    @property
    def type(self):
        return ConstraintParameterType.Enum

    def __repr__(self):
        return f"EnumParam({self.parameterId}: {self.value})"

    def to_dict(self):
        enumNames = {'direction': 'DimensionDirection', 
                     'halfSpace0': 'DimensionHalfSpace', 
                     'halfSpace1': 'DimensionHalfSpace',
                     'halfSpace2': 'DimensionHalfSpace',
                     'halfSpace3': 'DimensionHalfSpace',
                     'alignment': 'DimensionAlignment',
                     'projectionType': 'SketchProjectionType',
                     'radiusDisplay': 'RadiusDisplay',
                     'sketchToolType': 'SketchToolType'
                    }
        return {
            'message': {
                'enumName': enumNames[self.parameterId],
                'parameterId': self.parameterId,
                'value': self.value
            },
            'type': 145,
            'typeName': 'BTMParameterEnum'
        }

    @staticmethod
    def from_dict(param_dict):
        msg_dict = param_dict['message']
        return EnumParameter(
            *_get_parameter_common_attributes(param_dict),
            msg_dict['value'])


class BooleanParameter(ConstraintParameter):
    """Boolean parameter."""
    def __init__(self, parameterId, value):
        super(BooleanParameter, self).__init__(parameterId)
        self.value = value

    @property
    def type(self):
        return ConstraintParameterType.Boolean

    def __repr__(self):
        return f"BooleanParam({self.parameterId}: {self.value})"

    def to_dict(self):
        return {
            'message': {
                'parameterId': self.parameterId,
                'value': self.value
            },
            'type': 144,
            'typeName': 'BTMParameterBoolean'
        }

    @staticmethod
    def from_dict(param_dict):
        msg_dict = param_dict['message']
        return BooleanParameter(
            *_get_parameter_common_attributes(param_dict),
            msg_dict['value'])

_parameter_type_to_class = {
    ConstraintParameterType.LocalReference: LocalReferenceParameter,
    ConstraintParameterType.ExternalReference: ExternalReferenceParameter,
    ConstraintParameterType.Quantity: QuantityParameter,
    ConstraintParameterType.Enum: EnumParameter,
    ConstraintParameterType.Boolean: BooleanParameter
}


class Constraint(typing.NamedTuple):
    """This class represents a constraint.

    A constraint is represented by a given type, and a list of parameters,
    which specify the concrete effect of the constraint.
    """
    identifier: str
    constraint_type: ConstraintType
    parameters: typing.List[ConstraintParameter]

    def to_dict(self) -> dict:
        """Serializes this constraint as a dictionary compatible with the json representation."""
        return {
            'type': 2,
            'typeName': 'BTMSketchConstraint',
            'message': {
                'entityId': self.identifier,
                'constraintType': self.constraint_type.name.upper(),
                'parameters': [p.to_dict() for p in self.parameters]
            }
        }

    @staticmethod
    def from_dict(constr_dict):
        """Constructs a constraint from its dictionary representation."""
        msg_dict = constr_dict['message']

        constraint_type = string_to_constraint_type.get(
            msg_dict['constraintType'], ConstraintType.Unknown)
        parameters = [ConstraintParameter.from_dict(param)
                      for param in msg_dict['parameters']]
        return Constraint(msg_dict['entityId'], constraint_type, parameters)

    @property
    def type(self) -> ConstraintType:
        """Returns the type of this constraint."""
        return self.constraint_type

    def get_references(self) -> typing.List[str]:
        """Returns a list of the entities that are referred to by this constraint."""
        result = []

        for parameter in self.parameters:
            if isinstance(parameter, LocalReferenceParameter):
                result.append(parameter.value)
            elif isinstance(parameter, ExternalReferenceParameter):
                result.append('External')

        return result

__all__ = [
    'ConstraintType', 'Constraint', 'ConstraintParameter', 'ConstraintParameterType', 'GenericParameter',
    'LocalReferenceParameter', 'ExternalReferenceParameter', 'QuantityParameter', 'EnumParameter', 'BooleanParameter',
    'DirectionValue', 'HalfSpaceValue', 'AlignmentValue', 'BooleanValue']
