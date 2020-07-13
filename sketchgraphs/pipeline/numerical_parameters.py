"""This module implements functionality to handle numerical parameters in the sketch.

"""

import math
import re

import numpy as np
from sklearn.cluster import KMeans

_IMPLICIT_MUL_PATTERN = re.compile(r"(?<=\))\s*(?=\w+) | (?<=\d)\s*(?=[A-Z]+)", re.X)


##### Constraint handling

METER_CONVERSIONS = {
    'METER': 1,
    'METERS': 1,
    'M': 1,
    'MILLIMETER': 1e-3,
    'MILLIMETERS': 1e-3,
    'MM': 1e-3,
    'CM': 1e-2,
    'CENTIMETER': 1e-2,
    'CENTIMETERS': 1e-2,
    'INCHES': 0.0254,
    'INCH': 0.0254,
    'IN': 0.0254,
    'FOOT': 0.3048,
    'FEET': 0.048,
    'FT': 0.3048,
    'YD': 0.9144
}

DEGREE_CONVERSIONS = {
    'DEG': 1,
    'DEGREE': 1,
    'RADIAN': 180/np.pi,
    'RAD': 180/np.pi
}


def normalize_expression(expression, parameter_id):
    """Converts a numerical expression into a normalized form.

    Parameters
    ----------
    expression : str
        A string representing a quantity parameter value
    parameter_id : str
        A parameterId string; must be one of 'angle' or 'length'

    Returns
    -------
    norm_expression : str or None
        Normalized expression string if successful, or None otherwise.
    """
    expression = expression.upper().strip()

    if parameter_id == 'angle':
        conversions = DEGREE_CONVERSIONS
        default_unit = 'DEGREE'
    elif parameter_id == 'length':
        conversions = METER_CONVERSIONS
        default_unit = 'METER'
    else:
        raise ValueError('parameter_id must be one of angle or length')

    if '#' in expression or 'lookup' in expression:
        # variables in feature script are not supported
        return None

    # Adds implicit multiplication signs in order to be more easily parsed.
    expression = _IMPLICIT_MUL_PATTERN.sub('*', expression)

    try:
        value = eval(expression, {'PI': np.pi, 'SQRT': math.sqrt, 'TAN': math.tan}, conversions)
    except:
        return None

    value_str = np.format_float_positional(value, precision=4, trim='0', fractional=False)
    return '{} {}'.format(value_str, default_unit)


def make_quantization(values, num_points, scheme):
    """Find optimal centers for parameter via either uniform, K-means, or CDF-based K-means.

    Obtains a quantization scheme for the given values, according to a given strategy.
    Several schemes are supported, although we prefer the 'cdf' scheme, a hybrid which
    avoids some issues with large outliers in the datasets faced by other schemes.

    Parameters
    ----------
    values : np.array
        An array of values representing a sample of the values to quantize
    num_points : int
        Number of points to obtain in the dictionary
    scheme : str
        Indicates the quantization scheme to use, must be 'uniform', 'kmeans' or 'cdf'

    Returns
    -------
    np.array
        Array of quantization codes
    """

    if scheme == 'uniform':
        edges = np.linspace(np.min(values), np.max(values), num_points+1)
        return np.array([(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)])

    if scheme == 'kmeans':
        km = KMeans(n_clusters=num_points)
        km.fit(values.reshape(-1, 1))
        return np.sort(np.squeeze(km.cluster_centers_))

    if scheme == 'cdf':
        values, cdf = make_unique_cdf(values)
        cdf_centers = make_quantization(cdf, num_points, 'kmeans')
        return np.interp(cdf_centers, cdf, values)

    schemes = ['uniform', 'kmeans', 'cdf']
    raise ValueError("scheme must be one of " + str(schemes))


def make_unique_cdf(arr):
    """Return 'collapsed' cdf of arr (identical/close arr vals all have same cdf point).

    Parameters
    ----------
    arr: array of parameter values

    Returns
    -------
    sorted_arr: sorted copy of arr.
    cdf: collapsed cdf of arr.
    """
    cdf = np.linspace(0, 1, len(arr))
    sorted_arr = np.sort(arr)
    last_unique_idx = 0
    for idx, arr_val in enumerate(np.append(sorted_arr, [np.inf])):
        if not np.isclose(arr_val, sorted_arr[last_unique_idx]):
            cdf[last_unique_idx:idx] = np.mean(cdf[last_unique_idx:idx])
            last_unique_idx = idx
    return sorted_arr, cdf
