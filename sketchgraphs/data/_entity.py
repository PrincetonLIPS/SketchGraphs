"""Representation and parsing of entities in onshape json.
"""
import abc
import enum
import math
import typing

import numpy as np

# pylint: disable=invalid-name, too-many-arguments, too-many-return-statements, too-many-instance-attributes


class EntityType(enum.IntEnum):
    """Enumeration indicating the type of entity represented.
    """
    Point = 0
    Line = 1
    Circle = 2
    Ellipse = 3
    Spline = 4
    Conic = 5
    Arc = 6
    External = 7
    Stop = 8
    Unknown = 9


class SubnodeType(enum.IntEnum):
    SN_Start = 101
    SN_End = 102
    SN_Center = 103


def inspect_entity_type(entity_dict) -> EntityType:
    """Inspects the entity dictionary and determines the corresponding entity type."""

    if entity_dict['typeName'] == 'BTMSketchPoint':
        return EntityType.Point

    if 'geometry' not in entity_dict['message']:
        # Non-geometry entity (e.g. Text), simply treat as unknown
        return EntityType.Unknown

    geom_type = entity_dict['message']['geometry']['typeName']
    options = [
        ('Line', EntityType.Line),
        ('Circle', EntityType.Circle),
        ('Ellipse', EntityType.Ellipse),
        ('Spline', EntityType.Spline),
        ('Conic', EntityType.Conic)]
    for option, entity_type in options:
        if option in geom_type:
            break
    else:
        return EntityType.Unknown

    if option == 'Circle':
        if entity_dict['typeName'] == 'BTMSketchCurveSegment':
            return EntityType.Arc
    if option == 'Ellipse':
        if entity_dict['typeName'] == 'BTMSketchCurveSegment':
            # really should be EllipseArc here
            return EntityType.Unknown

    if option == 'Spline':
        if entity_dict['message']['geometry']['type'] == 117:
            return EntityType.Spline
        # BTCurveGeometryInterpolatedSpline
        return EntityType.Unknown

    return entity_type


def inspect_entity_type_from_info(ent_info) -> EntityType:
    """Inspects the entity info dictionary and determines the corresponding entity type."""
    options = {
        'point': EntityType.Point,
        'lineSegment': EntityType.Line,
        'circle': EntityType.Circle,
        'arc': EntityType.Arc}

    if ent_info['entityType'] in options:
        return options[ent_info['entityType']]
    else:
        raise ValueError('Entity type not supported.')


def _get_entity_common_attributes(ent_dict):
    msg_dict = ent_dict['message']
    return msg_dict['entityId'], bool(msg_dict['isConstruction'])


def _get_linestyle(entity):
    return '--' if entity.isConstruction else '-'


class Entity(abc.ABC):
    """Abstract class representing a geometry entity.

    This abstract class represents a geometric entity in a sketch.
    It is identified by its entity id, a unique string within the sketch.
    """
    entityId: str
    isConstruction: bool

    bool_ids = ['isConstruction']

    def __init__(self, entityId, isConstruction):
        self.entityId = entityId
        self.isConstruction = isConstruction

    @property
    @abc.abstractmethod
    def type(self) -> EntityType:
        """Get the concrete type of the underlying entity."""

    def get_subnode_ids(self):
        """Returns the subentity IDs of this entity instance."""
        return ()

    @staticmethod
    def get_subnode_types():
        """Returns list of SubnodeType enums for this entity."""
        return []

    @classmethod
    def get_subnode_type_names(cls):
        return [t.name for t in cls.get_subnode_types()]

    @abc.abstractmethod
    def to_dict(self) -> dict:
        """Obtains a serialized representation of this entity as a dictionary.

        The returned dictionary should be compatible with the json representation from onshape.
        """

    @staticmethod
    def from_dict(ent_dict):
        """Creates an entity from its json representation as a dictionary.

        Parameters
        ----------
        ent_dict : dict
            The dictionary containing the json data representing the entity.

        Returns
        -------
        Entity
            An entity of the appropriate type according to the dictionar data.
        """
        entity_type = inspect_entity_type(ent_dict)

        EntityClass = ENTITY_TYPE_TO_CLASS.get(entity_type, GenericEntity)
        return EntityClass.from_dict(ent_dict)

    @staticmethod
    def from_info(ent_info):
        """Creates an entity from its `info` dict representation.

        Parameters
        ----------
        ent_info: the dictionary containing the data representing the entity returned by sketch information call.
        """
        entity_type = inspect_entity_type_from_info(ent_info)

        EntityClass = ENTITY_TYPE_TO_CLASS.get(entity_type, GenericEntity)
        return EntityClass.from_info(ent_info)


class GenericEntity(Entity):
    """A generic unstructured entity.

    This class is used to capture entities that we do not fully support yet.
    """
    data: dict

    float_ids = ()
    bool_ids = ()

    def __init__(self, entityData):
        entityId = entityData['message']['entityId']
        isConstruction = entityData['message']['isConstruction']
        super(GenericEntity, self).__init__(entityId, isConstruction)
        self._type = inspect_entity_type(entityData)
        self.data = entityData

    @property
    def type(self):
        return self._type

    def __repr__(self):
        return f"Generic [{self.entityId}] type {self._type}"

    def to_dict(self):
        return self.data

    @staticmethod
    def from_dict(ent_dict):
        return GenericEntity(ent_dict)


class Point(Entity):
    """Point Entity."""

    float_ids = ['x', 'y']
    bool_ids = Entity.bool_ids

    def __init__(self, entityId, isConstruction=False, x=0, y=0):
        super(Point, self).__init__(entityId, isConstruction)
        self.x = x
        self.y = y

    @property
    def type(self):
        return EntityType.Point

    def to_dict(self):
        return {
            'type': 158,
            'typeName': 'BTMSketchPoint',
            'message': {
                'entityId': self.entityId,
                'isConstruction': self.isConstruction,
                'x': self.x,
                'y': self.y
            }
        }

    def __iter__(self):
        return iter((self.x, self.y))

    def __getitem__(self, idx):
        if idx == 0:
            return self.x
        elif idx == 1:
            return self.y
        else:
            raise IndexError()

    @staticmethod
    def from_dict(ent_dict):
        if ent_dict['typeName'] != 'BTMSketchPoint':
            raise ValueError('invalid dictionary for entity type Point')

        msg_dict = ent_dict['message']
        return Point(*_get_entity_common_attributes(ent_dict),
                     float(msg_dict['x']), float(msg_dict['y']))

    @staticmethod
    def from_info(ent_info):
        return Point(ent_info['id'],
                     bool(ent_info.get('isConstruction', False)),
                     float(ent_info['point'][0]),
                     float(ent_info['point'][1]))

    def __repr__(self):
        return f"Point [{self.entityId}] ({self.x, self.y})"


class Line(Entity):
    """Line Entity."""

    dirX: float
    dirY: float
    pntX: float
    pntY: float
    startParam: float
    endParam: float

    startPointId: str
    endPointId: str

    float_ids = ['dirX', 'dirY', 'pntX', 'pntY', 'startParam', 'endParam']
    bool_ids = Entity.bool_ids

    def __init__(self, entityId, isConstruction=False, pntX=0, pntY=0, dirX=1, dirY=0, startParam=-0.5, endParam=0.5):
        # TODO: address these default parameter values
        super(Line, self).__init__(entityId, isConstruction)
        self.dirX = dirX
        self.dirY = dirY
        self.pntX = pntX
        self.pntY = pntY
        self.startParam = startParam
        self.endParam = endParam

    @property
    def type(self):
        return EntityType.Line

    def get_subnode_ids(self):
        return (self.entityId + '.start', self.entityId + '.end')

    @staticmethod
    def get_subnode_types():
        return (SubnodeType.SN_Start, SubnodeType.SN_End)

    def to_dict(self):
        return {
            'message': {
                'entityId': self.entityId,
                'startPointId': self.entityId + '.start',
                'endPointId': self.entityId + '.end',
                'isConstruction': self.isConstruction,
                'startParam': self.startParam,
                'endParam': self.endParam,
                'geometry': {
                    'message': {
                        'dirX': self.dirX,
                        'dirY': self.dirY,
                        'pntX': self.pntX,
                        'pntY': self.pntY,
                    },
                    'type': 117,
                    'typeName': 'BTCurveGeometryLine',
                },
            },
            'type': 155,
            'typeName': 'BTMSketchCurveSegment',
        }

    @staticmethod
    def from_dict(ent_dict):
        msg_dict = ent_dict['message']
        geom_dict = msg_dict['geometry']['message']

        return Line(
            *_get_entity_common_attributes(ent_dict),
            float(geom_dict['pntX']),
            float(geom_dict['pntY']),
            float(geom_dict['dirX']),
            float(geom_dict['dirY']),
            float(msg_dict['startParam']),
            float(msg_dict['endParam']))

    @staticmethod
    def from_info(ent_info):
        startPoint = np.array(ent_info['startPoint'])
        endPoint = np.array(ent_info['endPoint'])
        vec = endPoint - startPoint
        length = np.linalg.norm(vec)
        if length == 0:
            dirX, dirY = 1, 0
        else:
            dirX, dirY = vec / length
        pntX = (startPoint[0] + endPoint[0]) / 2
        pntY = (startPoint[1] + endPoint[1]) / 2
        startParam = -1 * length / 2
        endParam = -1 * startParam

        return Line(
            ent_info['id'], bool(ent_info.get('isConstruction', False)), 
            pntX, pntY, dirX, dirY, startParam, endParam)

    @staticmethod
    def from_points(start_point, end_point):
        """Returns a Line instance based on start and end points provided.
        """
        ent_info = {'id': '',
                    'startPoint': [start_point.x, start_point.y],
                    'endPoint': [end_point.x, end_point.y]}
        return Line.from_info(ent_info)

    @property
    def start_point(self):
        """Returns a tuple representing the start location of the line."""
        startX = self.pntX + self.startParam * self.dirX
        startY = self.pntY + self.startParam * self.dirY
        return np.array([startX, startY])

    @property
    def end_point(self):
        """Returns a tuple representing the end location of the line."""
        endX = self.pntX + self.endParam * self.dirX
        endY = self.pntY + self.endParam * self.dirY
        return np.array([endX, endY])

    def __repr__(self):
        return f"Line [{self.entityId}] p({self.pntX}, {self.pntY}) d({self.dirX}, {self.dirY}) param({self.startParam}, {self.endParam})"  # pylint: disable=line-too-long


class Circle(Entity):
    """Circle Entity."""

    xCenter: float
    yCenter: float
    xDir: float
    yDir: float
    radius: float
    clockwise: bool

    float_ids = ['xCenter', 'yCenter', 'xDir', 'yDir', 'radius']
    bool_ids = Entity.bool_ids + ['clockwise']

    def __init__(self, entityId, isConstruction=False, xCenter=0, yCenter=0, xDir=1, yDir=0, radius=1, clockwise=False):
        super(Circle, self).__init__(entityId, isConstruction)
        self.xCenter = xCenter
        self.yCenter = yCenter
        self.xDir = xDir
        self.yDir = yDir
        self.radius = radius
        self.clockwise = clockwise

    @property
    def type(self):
        return EntityType.Circle

    def get_subnode_ids(self):
        return (self.entityId + '.center',)

    @staticmethod
    def get_subnode_types():
        return (SubnodeType.SN_Center,)

    def to_dict(self):
        return {
            'message': {
                'entityId': self.entityId,
                'centerId': self.entityId + '.center',
                'isConstruction': self.isConstruction,
                'geometry': {
                    'message': {
                        'xCenter': self.xCenter,
                        'yCenter': self.yCenter,
                        'xDir': self.xDir,
                        'yDir': self.yDir,
                        'radius': self.radius,
                        'clockwise': self.clockwise
                    },
                    'type': 115,
                    'typeName': 'BTCurveGeometryCircle',
                },
            },
            'type': 4,
            'typeName': 'BTMSketchCurve',
        }

    @property
    def center_point(self):
        return np.array([self.xCenter, self.yCenter])

    @staticmethod
    def from_dict(ent_dict):
        msg_dict = ent_dict['message']
        geom_dict = msg_dict['geometry']['message']

        return Circle(
            *_get_entity_common_attributes(ent_dict),
            float(geom_dict['xCenter']),
            float(geom_dict['yCenter']),
            float(geom_dict['xDir']),
            float(geom_dict['yDir']),
            float(geom_dict['radius']),
            bool(geom_dict['clockwise']))

    @staticmethod
    def from_info(ent_info):
        xCenter, yCenter = ent_info['center']
        xDir, yDir = 1.0, 0.0
        radius = ent_info['radius']
        clockwise = bool(ent_info.get('clockwise', False))

        return Circle(ent_info['id'],
                      bool(ent_info.get('isConstruction', False)),
                      xCenter, yCenter, xDir, yDir,
                      radius, clockwise)


    def __repr__(self):
        return (f"Circle [{self.entityId}] c({self.xCenter}, {self.yCenter}) " +
                f"d({self.xDir}, {self.yDir}) r({self.radius}) " +
                f"{'clockwise' if self.clockwise else 'anti-clockwise'}")


class Arc(Entity):
    """Arc entity"""

    xCenter: float
    yCenter: float
    xDir: float
    yDir: float
    radius: float
    clockwise: bool
    startParam: float
    endParam: float

    float_ids = ['xCenter', 'yCenter', 'xDir', 'yDir', 'radius', 'startParam', 'endParam']
    bool_ids = Circle.bool_ids

    def __init__(self, entityId, isConstruction=False,
                 xCenter=0, yCenter=0, xDir=1, yDir=0,
                 radius=1, clockwise=False, startParam=-0.5, endParam=0.5):
        super(Arc, self).__init__(entityId, isConstruction)
        self.xCenter = xCenter
        self.yCenter = yCenter
        self.xDir = xDir
        self.yDir = yDir
        self.radius = radius
        self.clockwise = clockwise
        self.startParam = startParam
        self.endParam = endParam

    @property
    def type(self):
        return EntityType.Arc

    def get_subnode_ids(self):
        return (self.entityId + '.center', self.entityId + '.start', self.entityId + '.end')

    @staticmethod
    def get_subnode_types():
        return (SubnodeType.SN_Center, SubnodeType.SN_Start, SubnodeType.SN_End)

    def to_dict(self):
        return {
            'message': {
                'centerId': self.entityId + '.center',
                'entityId': self.entityId,
                'isConstruction': self.isConstruction,
                'startParam': self.startParam,
                'endParam': self.endParam,
                'startPointId': self.entityId + '.start',
                'endPointId': self.entityId + '.end',
                'geometry': {
                    'type': 115,
                    'typeName': 'BTCurveGeometryCircle',
                    'message': {
                        'xCenter': self.xCenter,
                        'yCenter': self.yCenter,
                        'xDir': self.xDir,
                        'yDir': self.yDir,
                        'radius': self.radius,
                        'clockwise': self.clockwise
                    }
                }
            },
            'type': 155,
            'typeName': 'BTMSketchCurveSegment',
        }


    @staticmethod
    def from_dict(ent_dict):
        msg_dict = ent_dict['message']
        geom_dict = msg_dict['geometry']['message']

        return Arc(
            *_get_entity_common_attributes(ent_dict),
            float(geom_dict['xCenter']),
            float(geom_dict['yCenter']),
            float(geom_dict['xDir']),
            float(geom_dict['yDir']),
            float(geom_dict['radius']),
            bool(geom_dict['clockwise']),
            float(msg_dict['startParam']),
            float(msg_dict['endParam']))

    @staticmethod
    def from_info(ent_info):
        xCenter, yCenter = ent_info['center']
        xDir, yDir = 1.0, 0.0
        radius = ent_info['radius']
        clockwise = bool(ent_info.get('clockwise', False))
        startVec = np.array(ent_info['startPoint']) - np.array(ent_info['center'])
        endVec = np.array(ent_info['endPoint']) - np.array(ent_info['center'])
        startParam = math.atan2(startVec[1], startVec[0])
        endParam = math.atan2(endVec[1], endVec[0])
        if clockwise:
            # Convert to counterclockwise
            tmp_start = startParam
            startParam = endParam
            endParam = tmp_start

        return Arc(ent_info['id'],
                   bool(ent_info.get('isConstruction', False)),
                   xCenter, yCenter, xDir, yDir,
                   radius, False, startParam, endParam)

    def _point_at_angle_offset(self, angle_offset):
        angle = math.atan2(self.yDir, self.xDir)
        if self.clockwise:
            angle_offset *= -1
        angle_start = angle + angle_offset
        xStart = self.xCenter + (math.cos(angle_start) * self.radius)
        yStart = self.yCenter + (math.sin(angle_start) * self.radius)
        return np.array([xStart, yStart])

    @property
    def start_point(self):
        """Returns tuple representing coordinates of arc start point."""
        return self._point_at_angle_offset(self.startParam)

    @property
    def end_point(self):
        """Returns tuple representing coordinates of arc end point."""
        return self._point_at_angle_offset(self.endParam)

    @property
    def mid_point(self):
        """Returns tuple representing coordinates of arc midpoint."""
        start_param = self.startParam % (2*np.pi)
        end_param = self.endParam % (2*np.pi)
        if start_param > end_param:
            end_param += 2*np.pi
        mid_param = np.mean([start_param, end_param])
        return self._point_at_angle_offset(mid_param)

    @property
    def center_point(self):
        return np.array([self.xCenter, self.yCenter])

    def __repr__(self):
        return (f"Arc [{self.entityId}] c({self.xCenter}, {self.yCenter}) " +
                f"d({self.xDir}, {self.yDir}) r({self.radius}) " +
                f"{'clockwise' if self.clockwise else 'anti-clockwise'}")


class Spline(Entity):
    """B-spline entity """

    degree: int
    isPeriodic: bool
    isRational: bool
    controlPoints: np.array
    knots: np.array
    startParam: typing.Optional[float]
    endParam: typing.Optional[float]

    def __init__(self, entityId, isConstruction, degree, controlPoints,
                 knots, isPeriodic, isRational, startParam, endParam):
        super(Spline, self).__init__(entityId, isConstruction)

        if len(knots) != len(controlPoints) + degree + 1:
            raise ValueError('incompatible B-spline specification')

        self.degree = degree
        self.isPeriodic = isPeriodic
        self.isRational = isRational
        self.controlPoints = controlPoints
        self.knots = knots
        self.startParam = startParam
        self.endParam = endParam

    def to_dict(self):
        return {
            'message': {
                'entityId': self.entityId,
                'isConstruction': self.isConstruction,
                'geometry': {
                    'message': {
                        'controlPointCount': self.controlPoints.shape[0],
                        'controlPoints': self.controlPoints.reshape(-1).tolist(),
                        'knots': self.knots.tolist(),
                        'degree': self.degree,
                        'isPeriodic': self.isPeriodic,
                        'isRational': self.isRational,
                    },
                    'type': 117,
                    'typeName': 'BTCurveGeometrySpline',
                },
                **({
                    'startParam': self.startParam,
                    'endParam': self.endParam
                } if self.startParam is not None else {})
            },
            'type': 4 if self.isPeriodic else 155,
            'typeName': 'BTMSketchCurve' if self.isPeriodic else 'BTMSketchCurveSegment'
        }

    @staticmethod
    def from_dict(ent_dict):
        msg_dict = ent_dict['message']
        geom_dict = msg_dict['geometry']['message']
        controlPoints = np.array(geom_dict['controlPoints'], dtype=np.float).reshape(-1, 2)
        knots = np.array(geom_dict['knots'], dtype=np.float)

        return Spline(
            *_get_entity_common_attributes(ent_dict),
            int(geom_dict['degree']),
            controlPoints,
            knots,
            bool(geom_dict['isPeriodic']),
            bool(geom_dict['isRational']),
            msg_dict.get('startParam'),
            msg_dict.get('endParam'))

    @property
    def type(self):
        return EntityType.Spline

    def get_subIds(self):
        # TODO: implement this for splines
        return []

    def __repr__(self):
        return f"Spline [{self.entityId}] k({self.knots.shape[0]}) p({self.controlPoints.shape[0]}) d({self.degree})"  # pylint:disable=line-too-long


class Ellipse(Entity):
    """Ellipse entity."""

    xCenter: float
    yCenter: float
    xDir: float
    yDir: float
    radius: float
    minorRadius: float
    clockwise: bool

    float_ids = ['xCenter', 'yCenter', 'xDir', 'yDir', 'radius', 'minorRadius']
    bool_ids = Entity.bool_ids + ['clockwise']

    def __init__(self, entityId, isConstruction, xCenter, yCenter,
                 xDir, yDir, radius, minorRadius, clockwise):
        super(Ellipse, self).__init__(entityId, isConstruction)

        self.xCenter = xCenter
        self.yCenter = yCenter
        self.xDir = xDir
        self.yDir = yDir
        self.radius = radius
        self.minorRadius = minorRadius
        self.clockwise = clockwise

    def to_dict(self):
        return {
            'message': {
                'entityId': self.entityId,
                'isConstruction': self.isConstruction,
                'geometry': {
                    'type': 115,
                    'typeName': 'BTCurveGeometryEllipse',
                    'message': {
                        'xCenter': self.xCenter,
                        'yCenter': self.yCenter,
                        'xDir': self.xDir,
                        'yDir': self.yDir,
                        'radius': self.radius,
                        'minorRadius': self.minorRadius,
                        'clockwise': self.clockwise
                    }
                }
            },
            'type': 4,
            'typeName': 'BTMSketchCurve',
        }

    @staticmethod
    def from_dict(ent_dict):
        msg_dict = ent_dict['message']
        geom_dict = msg_dict['geometry']['message']

        return Ellipse(
            *_get_entity_common_attributes(ent_dict),
            float(geom_dict['xCenter']),
            float(geom_dict['yCenter']),
            float(geom_dict['xDir']),
            float(geom_dict['yDir']),
            float(geom_dict['radius']),
            float(geom_dict['minorRadius']),
            bool(geom_dict['clockwise']))

    @property
    def type(self):
        return EntityType.Ellipse

    def __repr__(self):
        return (f"Ellipse [{self.entityId}] c({self.xCenter}, {self.yCenter}) " +
                f"d({self.xDir}, {self.yDir}) r({self.radius}, {self.minorRadius}) " +
                f"{'clockwise' if self.clockwise else 'anti-clockwise'}")


ENTITY_TYPE_TO_CLASS = {
    EntityType.Point: Point,
    EntityType.Line: Line,
    EntityType.Circle: Circle,
    EntityType.Arc: Arc,
    EntityType.Spline: Spline,
    EntityType.Ellipse: Ellipse,
}


def string_to_entity_class(ent_name):
    return ENTITY_TYPE_TO_CLASS[EntityType[ent_name]]


__all__ = [
    'EntityType', 'SubnodeType', 'Entity', 'GenericEntity', 'Point', 'Line',
    'Circle', 'Arc', 'Spline', 'Ellipse', 'ENTITY_TYPE_TO_CLASS'
]
