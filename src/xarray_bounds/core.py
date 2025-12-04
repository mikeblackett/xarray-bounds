from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd
import xarray as xr

from xarray_bounds.helpers import (
    validate_interval_closed,
    validate_interval_label,
)
from xarray_bounds.options import OPTIONS
from xarray_bounds.types import (
    ClosedSide,
    IntervalClosed,
    IntervalLabel,
    LabelSide,
)
from xarray_bounds.utilities import infer_interval

__all__ = [
    'infer_bounds',
    'bounds_to_interval',
    'interval_to_bounds',
]


def infer_bounds(
    obj: xr.DataArray,
    *,
    label: IntervalLabel | None = None,
    closed: IntervalClosed | None = None,
) -> xr.DataArray:
    """Infer bounds for a 1D indexed coordinate.

    The index must be monotonic increasing and regularly spaced.

    Parameters
    ----------
    obj : DataArray
        The indexed coordinate to infer bounds from.
    label : Literal['left', 'middle', 'right'], optional
        Which bin edge or midpoint the index labels. If None, defaults to 'left'
        if `closed` is 'left' and 'right' if `closed` is 'right'. Otherwise, defaults
        to 'left'.
    closed : Literal['left', 'right'], optional
        The closed side(s) of the interval.
        If None, defaults to 'left' if `label` is 'left' and 'right' if `label` is 'right'.

    Returns
    -------
    DataArray
        An array with shape ``(n, 2)`` representing the bounds.

    Raises
    ------
    ValueError
        If the array is not 1D.
    TypeError
        If the array does not have compatible index.
    ValueError
        If a regular frequency cannot be inferred from the index.
    """
    try:
        index = obj.to_index()
    except ValueError:
        raise ValueError(
            'Bounds inference is only supported for 1-dimensional coordinates.'
        )

    obj = obj.copy()

    label = validate_interval_label(label=label, default=closed)
    closed = validate_interval_closed(closed=closed, default=label)

    interval = infer_interval(
        index=index,
        label=LabelSide(label),
        closed=ClosedSide(closed),
    )

    data = np.stack(arrays=(interval.left, interval.right), axis=1)

    dim = index.name
    coord = obj.name or dim
    bounds = f'{coord}_{OPTIONS["bounds_dim"]}'
    obj.attrs['bounds'] = bounds

    return xr.DataArray(
        data=data,
        coords={coord: obj},
        dims=(dim, OPTIONS['bounds_dim']),
        name=bounds,
        attrs={'closed': interval.closed},
    )


def interval_to_bounds(
    interval: pd.IntervalIndex,
    *,
    label: IntervalLabel | None = None,
    dim: Hashable | None = None,
    name: Hashable | None = None,
) -> xr.DataArray:
    """Convert a :py:class:`pandas.IntervalIndex` into a boundary coordinate.

    Parameters
    ----------
    interval : pd.IntervalIndex
        The interval index to convert.
    label : Literal['left', 'middle', 'right'], optional
        Which point the index coordinate should label. If None, defaults to the
        closed side of the interval.
    dim : Hashable, optional
        The dimension name for the axis coordinate. If None, defaults to ``interval.name``.
    name : Hashable, optional
        The variable name for the axis coordinate. Defaults to the computed value of `dim`.
        This is only needed if you want the variable name to be different from the dimension name.

    Returns
    -------
    DataArray
        A boundary variable with shape ``(n, 2)``.
    """
    if interval.closed not in ClosedSide:
        raise ValueError(
            f'IntervalIndex has unsupported closed attribute: {interval.closed}. '
            f'Must be one of {list(ClosedSide)}.'
        )

    dim = dim or interval.name

    if dim is None:
        raise ValueError(
            'If the interval has no name, a dimension name must be provided.'
        )

    if name is None:
        name = dim

    label = validate_interval_label(label=label, default=interval.closed)

    bounds = f'{name}_{OPTIONS["bounds_dim"]}'

    data = np.stack(arrays=(interval.left, interval.right), axis=1)
    axis = (
        pd.Index(
            getattr(interval, 'mid' if label == 'middle' else label),
            name=dim,
        )
        .to_series()
        .to_xarray()
        .assign_attrs(bounds=bounds)
        .cf.guess_coord_axis()
    )

    return xr.DataArray(
        data=data,
        dims=(dim, OPTIONS['bounds_dim']),
        coords={name: axis},
        name=bounds,
        attrs={'closed': interval.closed},
    )


def bounds_to_interval(obj: xr.DataArray) -> pd.IntervalIndex:
    """Convert a boundary coordinate into a :py:class:`pandas.IntervalIndex`.

    Parameters
    ----------
    obj : DataArray
        2-dimensional boundary variable to convert.

    Returns
    -------
    pd.IntervalIndex
        A pandas interval index representing the bounds.
    """
    if obj.ndim != 2:
        raise ValueError(f'bounds must be a 2D DataArray, got {obj.ndim}D.')

    if obj.dims[1] != OPTIONS['bounds_dim']:
        raise ValueError(
            f'bounds must have a second dimension named {OPTIONS["bounds_dim"]}.'
        )

    closed = ClosedSide(obj.attrs.get('closed', 'left'))
    dim = obj.dims[0]

    return pd.IntervalIndex.from_arrays(
        *obj.transpose(),
        closed=closed.value,
        name=dim,
    )
