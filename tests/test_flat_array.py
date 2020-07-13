#pylint: disable=missing-module-docstring,missing-function-docstring

import itertools

import numpy as np

from sketchgraphs.data import flat_array


def test_flat_array_of_int():
    x = [2, 3, 4, 5]

    x_flat = flat_array.save_list_flat(x)
    x_array = flat_array.FlatSerializedArray.from_flat_array(x_flat)

    assert len(x_array) == len(x)

    for i, j in itertools.zip_longest(x, x_array):
        assert i == j


def test_flat_dictionary():
    x = [2, 3, 4, 5]
    y = np.array([3, 5])
    z = ["A", "python", "list"]

    x_flat = flat_array.save_list_flat(x)

    dict_flat = flat_array.pack_dictionary_flat({
        'x': x_flat,
        'y': y,
        'z': z
    })

    result = flat_array.load_dictionary_flat(dict_flat)

    assert isinstance(result['x'], flat_array.FlatSerializedArray)
    assert len(result['x']) == len(x)

    assert result['z'] == z
    assert all(result['y'] == y)
