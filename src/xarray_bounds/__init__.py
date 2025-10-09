"""
This package provides xarray accessors for adding CF-compliant bounds
coordinates to xarray objects. Bounds can be added for coordinates that label
a valid CF axis (i.e. any of ``{T, X, Y, Z}``). The coordinate variables must be
monotonic indexed dimension coordinates. The coordinates can be referenced by
their dimension name or by any name understood by the ``cf-xarray`` package.

This module will undergo significant breaking changes when xarray adds support
for ``CFIntervalIndex``. See https://github.com/pydata/xarray/pull/10296 for details.
"""

from xarray_bounds.accessors import DataArrayBounds, DatasetBounds
from xarray_bounds import helpers

__all__ = [
    'DataArrayBounds',
    'DatasetBounds',
    'helpers',
]
