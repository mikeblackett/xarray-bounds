"""
This package provides xarray accessors for adding CF-compliant bounds
coordinates to xarray objects. Bounds can be added for coordinates that label
a valid CF axis (i.e. any of ``{T, X, Y, Z}``). The coordinate variables must be
monotonic indexed dimension coordinates. The coordinates can be referenced by
their dimension name or by any name understood by the ``cf_xarray`` package.

This module will undergo significant breaking changes when xarray adds support
for ``CFIntervalIndex``. See https://github.com/pydata/xarray/pull/10296 for details.
"""

from xarray_bounds import datasets
from xarray_bounds._version import version as __version__
from xarray_bounds.accessors import (
    DataArrayBoundsAccessor,
    DatasetBoundsAccessor,
)
from xarray_bounds.core import (
    bounds_to_interval,
    infer_bounds,
    interval_to_bounds,
)
from xarray_bounds.options import set_options

__all__ = [
    '__version__',
    'bounds_to_interval',
    'DataArrayBoundsAccessor',
    'DatasetBoundsAccessor',
    'datasets',
    'infer_bounds',
    'interval_to_bounds',
    'set_options',
]
