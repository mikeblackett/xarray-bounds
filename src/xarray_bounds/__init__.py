"""
This module provides an xarray accessor for adding CF-compliant bounds
coordinates to xarray objects. Bounds can be added for coordinates that label
a valid CF axis (i.e. any of ``{T, X, Y, Z}``). The coordinate variables must be
monotonic indexed dimension coordinates. The coordinates can be referenced by
their coordinate name or by their corresponding CF axis key.
"""

from xarray_bounds.accessors import DataArrayBounds, DatasetBounds
from xarray_bounds.helpers import infer_bounds, datetime_to_interval

__all__ = [
    'DataArrayBounds',
    'DatasetBounds',
    'infer_bounds',
    'datetime_to_interval',
]
