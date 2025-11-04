from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any, cast, Hashable

import numpy as np
import pandas as pd
import xarray as xr

from xarray_bounds.helpers import (
    datetime_to_interval,
    index_to_interval,
)
from xarray_bounds.types import (
    ClosedSide,
    IntervalClosed,
    IntervalLabel,
    LabelSide,
    is_multi_index,
    is_datetime_index,
)
from xarray_bounds.options import OPTIONS


def infer_interval(
    index: pd.Index,
    *,
    label: LabelSide = LabelSide.LEFT,
    closed: ClosedSide = ClosedSide.LEFT,
    name: Hashable | None = None,
) -> pd.IntervalIndex:
    """Infer an interval index from a regularly spaced pandas index.

    If the index is a datetime index, it should have a regular (inferrable)
    frequency.

    Parameters
    ----------
    index : pd.Index
        The index to infer bounds for.
    label : Literal['left', 'middle', 'right'] | None, optional
        Which bin edge or midpoint the index labels.
    closed : Literal['left', 'right'] | None, optional
        The closed side of the interval.
    name : Hashable | None, optional
        The name of the interval index. If None, the name of the index is used.

    Returns
    -------
    pd.IntervalIndex
        The interval index representing the bounds.
    """
    if is_multi_index(index):
        raise ValueError('MultiIndex is not supported.')

    name = name or index.name

    if is_datetime_index(index):
        return datetime_to_interval(
            index=index, closed=closed, label=label, name=name
        )

    return index_to_interval(
        index=index, label=label, closed=closed, name=name
    )


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
        Which bin edge or midpoint the index labels.
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
    """
    try:
        index = obj.to_index()
    except ValueError:
        raise ValueError(
            'Bounds are only supported for 1-dimensional coordinates.'
        )

    label = label or obj.attrs.get('bounds_label', 'left')
    closed = closed or 'right' if label == 'right' else 'left'
    interval = infer_interval(
        index=index,
        closed=ClosedSide(closed),
        label=LabelSide(label),
    )

    data = np.stack(arrays=(interval.left, interval.right), axis=1)
    dim = index.name
    name = f'{dim}_{OPTIONS["bounds_dim"]}'
    coord = obj.assign_attrs(bounds=name)

    return xr.DataArray(
        data=data,
        coords={dim: coord},
        dims=(dim, OPTIONS['bounds_dim']),
        name=name,
        attrs={'closed': interval.closed},
    )


def bounds_to_index(obj: xr.DataArray) -> pd.IntervalIndex:
    """Convert a bounds array to a ``pandas.IntervalIndex``.

    Parameters
    ----------
    obj : DataArray
        The 2-dimensional bounds data array.

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


def mapping_or_kwargs[T](
    parg: Mapping[Any, T] | None,
    kwargs: Mapping[str, T],
    func_name: str,
) -> Mapping[Hashable, T]:
    """Return a mapping of arguments from either positional or keyword
    arguments."""
    if parg is None or parg == {}:
        return cast(Mapping[Hashable, T], kwargs)
    if kwargs:
        raise ValueError(
            f'cannot specify both keyword and positional arguments to {func_name}'
        )
    return parg
