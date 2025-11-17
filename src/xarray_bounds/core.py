from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd
import xarray as xr

from xarray_bounds.helpers import resolve_bounds_name
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
    closed : Literal['left', 'right', 'neither', 'both'], optional
        The closed side(s) of the interval.

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
            'Bounds are only supported for 1-dimensional coordinates.'
        )

    if label is None and closed is None:
        label = 'left'
        closed = 'left'
    if label is None:
        label = 'right' if closed == 'right' else 'left'
    if closed is None:
        closed = 'right' if label == 'right' else 'left'

    interval = infer_interval(
        index=index,
        label=LabelSide(label),
        closed=ClosedSide(closed),
    )

    data = np.stack(arrays=(interval.left, interval.right), axis=1)
    dim = index.name
    name = resolve_bounds_name(dim)
    coord = obj.assign_attrs(bounds=name)

    return xr.DataArray(
        data=data,
        coords={dim: coord},
        dims=(dim, OPTIONS['bounds_dim']),
        name=name,
        attrs={'closed': interval.closed},
    )


def interval_to_bounds(
    interval: pd.IntervalIndex,
    label: IntervalLabel = 'left',
    dim: Hashable | None = None,
) -> xr.DataArray:
    """Convert a :py:class:`pandas.IntervalIndex` into a boundary coordinate.

    Parameters
    ----------
    interval : pd.IntervalIndex
        The interval index to convert.
    label : Literal['left', 'middle', 'right'], default 'left'
        Which point the index coordinate should label.
    dim : Hashable, optional
        The dimension name for the coordinate. If None, uses the name of the
        interval index.

    Returns
    -------
    DataArray
        A 2-dimensional boundary variable representing the bounds.
    """
    data = np.stack(arrays=(interval.left, interval.right), axis=1)
    dim = dim or interval.name
    name = resolve_bounds_name(dim)

    return xr.DataArray(
        data=data,
        dims=(dim, OPTIONS['bounds_dim']),
        coords={
            dim: pd.Index(
                getattr(interval, 'mid' if label == 'middle' else label),
                name=dim,
            )
            .to_series()
            .to_xarray()
            .assign_attrs(bounds=name)
        },
        name=name,
        attrs={'closed': interval.closed},
    ).cf.guess_coord_axis()


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
    return pd.IntervalIndex.from_arrays(
        *obj.transpose(),
        closed=closed.value,
        name=obj.name,
    )
