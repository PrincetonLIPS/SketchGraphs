"""This module handles the quantization aspects of the dataset processing pipeline for the graph model. """

import enum
import collections

import numpy as np

from sketchgraphs.data import sketch as datalib
from . import _graph_info as graph_utils
from .target import TargetType


def _numerical_features(feature, params, feature_desc):
    for i, (param_name, param_type) in enumerate(feature_desc.items()):
        param_value = params.get(param_name)
        if isinstance(param_value, enum.IntEnum):
            param_value = param_value.name
        feature[i] = int(param_type[param_value])


def _get_param_features(angle_map, length_map):
    return collections.OrderedDict([
        (TargetType.EdgeAngle, collections.OrderedDict([
            ('aligned', datalib.BooleanValue),
            ('clockwise', datalib.BooleanValue),
            ('angle', angle_map)])),
        (TargetType.EdgeLength, collections.OrderedDict([
            ('direction', datalib.DirectionValue),
            ('length', length_map)])),
        (TargetType.EdgeDistance, collections.OrderedDict([
            ('direction', datalib.DirectionValue),
            ('halfSpace0', datalib.HalfSpaceValue),
            ('halfSpace1', datalib.HalfSpaceValue),
            ('length', length_map)])),
        (TargetType.EdgeDiameter, collections.OrderedDict([('length', length_map)])),
        (TargetType.EdgeRadius, collections.OrderedDict([('length', length_map)]))
    ])


class EdgeFeatureMapping:
    """Helper class for extracting numerical features from edges. """
    def __init__(self, angle=None, length=None):
        """Creates a new mapping, which uses the specified angle and length quantizers."""
        if angle is None:
            angle = QuantizationMap()

        if length is None:
            length = QuantizationMap()

        self.angle_map = angle
        self.length_map = length
        self._features_by_target = _get_param_features(angle, length)

    def state_dict(self):
        return {
            'angle': self.angle_map.state_dict(),
            'length': self.length_map.state_dict()
        }

    def load_state_dict(self, state):
        self.angle_map.load_state_dict(state['angle'])
        self.length_map.load_state_dict(state['length'])
        self._features_by_target = _get_param_features(self.angle_map, self.length_map)

    def sparse_numerical_features(self, edge_ops, target):
        """Creates sparse numeric features from the given edge ops for the given label. """
        index = []
        ops = []

        for i, e in enumerate(edge_ops):
            if TargetType.from_op(e) != target:
                continue
            index.append(i)
            ops.append(e)

        index = np.array(index, dtype=np.int64)
        features = self.numerical_features(ops, target)
        return graph_utils.SparseFeatureBatch(index, features)

    def numerical_features(self, ops, target) -> np.ndarray:
        feature_desc = self._features_by_target.get(target, {})
        dim = len(feature_desc)
        features = np.empty((len(ops), dim), dtype=np.int64)

        for i, op in enumerate(ops):
            _numerical_features(features[i], op.parameters, feature_desc)

        return features

    def features_from_index(self, indices, target):
        """Obtains categorical features from the given indices.

        Parameters
        ----------
        indices : sequence of integers
            A sequence of integers representing the index of each sub-feature
        target : `TargetType`
            The type of target that should be decoded from the indices

        Returns
        -------
        collections.OrderedDict
            An ordered dictionary containing the decoded features.
        """
        feature_desc = self._features_by_target[target]
        return collections.OrderedDict([
            (feature_name, feature_type(idx))
            for idx, (feature_name, feature_type) in zip(indices, feature_desc.items())
        ])


    def all_sparse_features(self, edge_ops):
        """Creates sparse features from the given edge ops for all supported labels.

        Parameters
        ----------
        edge_ops : List of EdgeOp
            The list of edge operations for which to extract the sparse features.

        Returns
        -------
        dict
            A dictionary with keys given by `self.supported_targets`, and values representing
            the corresponding sparse features in a `SparseFeatureBatch`.
        """
        return {
            k: self.sparse_numerical_features(edge_ops, k) for k in self.supported_targets
        }

    @property
    def feature_schema(self):
        """Gets a dictionary representing the schema for numerical features. """
        return self._features_by_target

    @property
    def feature_dimensions(self):
        return {
            k: collections.OrderedDict(
                (param_name, len(x)) for param_name, x in v.items())
            for k, v in self._features_by_target.items()
        }

    @property
    def supported_targets(self):
        return self._features_by_target.keys()


class QuantizationMap:
    """Utility class representing a quantization map.

    This mapping can be used by accessing the call and index operator of an instance
    (similarly to how an `Enum` behaves).

    If `qmap` is a quantization map, `qmap[value]` returns the index for the given
    value, and `qmap(index)` returns the value for the given index.
    """
    def __init__(self, values=None):
        if values is None:
            values = []

        self._values = list(values)
        self._values_to_idx = {
            v: i + 1 for i, v in enumerate(self._values)
        }

    @staticmethod
    def from_counter(counter, k):
        return QuantizationMap([k for k, v in counter.most_common(k)])

    def __call__(self, idx):
        if idx == 0:
            return 'OTHER'
        else:
            return self._values[idx - 1]

    def __getitem__(self, value):
        return self._values_to_idx.get(value, 0)

    def __len__(self):
        return len(self._values_to_idx) + 1

    def state_dict(self):
        return {
            'values': self._values
        }

    def load_state_dict(self, state):
        self._values = state['values']
        self._values_to_idx = {
            v: i + 1 for i, v in enumerate(self._values)
        }


def _convert_centers_to_edges(bin_centers):
    return {
        target_type: collections.OrderedDict(
            (field, 0.5 * (centers[1:] + centers[:-1]))
            for field, centers in bin_centers.get(entity_type, {}).items()
        ) for entity_type, target_type in _entity_label_to_target_type_dict.items()
    }


_entity_label_to_target_type_dict = {
    datalib.EntityType.Arc: TargetType.NodeArc,
    datalib.EntityType.Circle: TargetType.NodeCircle,
    datalib.EntityType.Line: TargetType.NodeLine,
    datalib.EntityType.Point: TargetType.NodePoint
}

_entity_label_from_target_type_dict = {v: k for k, v in _entity_label_to_target_type_dict.items()}


def _op_string_label_to_target_type(label):
    return _entity_label_to_target_type_dict.get(label, TargetType.NodeGeneric)


class EntityFeatureMapping:
    """Helper class for extracting sparse features from entities.

    In addition to features shared by all entity types (e.g. isConstruction),
    many entity types have their own features. In order to process them appropriately
    in batched fashion, they must be extracted into a set of sparse features.
    """
    def __init__(self, bin_centers_by_target=None):
        """Initialize a new mapping from the given bin definitions.

        Parameters
        ----------
        bin_centers_by_target : dict
            A dictionary specifying bin center locations for each target type (i.e. entity type)
        """
        self._bin_centers = bin_centers_by_target or {}
        self._features_by_target = _convert_centers_to_edges(self._bin_centers)

    def state_dict(self):
        return {
            'bin_centers': self._bin_centers
        }

    def load_state_dict(self, state_dict):
        self._bin_centers = state_dict['bin_centers']
        self._features_by_target = _convert_centers_to_edges(self._bin_centers)

    def _feature_length(self, target):
        return (len(self._features_by_target.get(target, {})) +
                int(target in (TargetType.NodeArc, TargetType.NodeCircle)) + 1)

    def _numerical_features(self, feature, params, target):
        # First feature is categorical isConstruction feature
        param_bin_edges = self._features_by_target.get(target, {})

        feature[0] = int(params['isConstruction'])
        offset = 1

        if target in (TargetType.NodeArc, TargetType.NodeCircle):
            offset += 1
            feature[1] = int(params['clockwise'])

        for i, (param_name, edges) in enumerate(param_bin_edges.items()):
            feature[i + offset] = int(np.searchsorted(edges, params[param_name]))

    def numerical_features(self, ops, target) -> np.ndarray:
        """Produces a dense array of numerical features.

        The operations in the `ops` array must all match the given target type.
        """
        features = np.empty((len(ops), self._feature_length(target)), dtype=np.int64)

        for i, op in enumerate(ops):
            self._numerical_features(features[i], op.parameters, target)

        return features

    def sparse_features_for_target(self, node_ops, target):
        """Produces sparse features for given target type.

        This function produces a sparse feature batch for the given sequence of ops
        and target type. The sparse features are emitted for each node in `node_ops` which
        matches the specified target type.
        """
        index = []
        ops = []

        for i, e in enumerate(node_ops):
            if _op_string_label_to_target_type(e.label) != target:
                continue
            index.append(i)
            ops.append(e)

        index = np.array(index, dtype=np.int64)
        features = self.numerical_features(ops, target)

        return graph_utils.SparseFeatureBatch(index, features)

    def all_sparse_features(self, node_ops):
        return {
            target: self.sparse_features_for_target(node_ops, target)
            for target in TargetType.numerical_node_types()
        }

    def features_from_index(self, index, target):
        """Obtains categorical features from the given indices. """
        features = collections.OrderedDict()

        features['isConstruction'] = bool(index[0])
        if target in (TargetType.NodeArc, TargetType.NodeCircle):
            features['clockwise'] = bool(index[1])
            offset = 2
        else:
            offset = 1

        target_entity = _entity_label_from_target_type_dict[target]

        for i, (param_name, centers) in enumerate(self._bin_centers.get(target_entity, {}).items()):
            features[param_name] = centers[index[i + offset]]

        return features

    @property
    def feature_dimensions(self):
        return {
            k: collections.OrderedDict(
                [('isConstruction', 2)] +
                ([('clockwise', 2)] if k in (TargetType.NodeArc, TargetType.NodeCircle) else []) +
                [(param_name, len(edges) + 1) for param_name, edges in v.items()])
            for k, v in self._features_by_target.items()
        }
