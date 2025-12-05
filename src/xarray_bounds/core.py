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
    """Infer the bounds for a 1D variable.

    Parameters
    ----------
    obj : DataArray
        A 1-dimensional variable with a monotonic and regularly spaced index.
    label : Literal['left', 'middle', 'right'], optional
        Which side or midpoint of the interval the index labels.
        If None, defaults to the value of ``closed`` if provided, otherwise defaults to 'left'.
    closed : Literal['left', 'right'], optional
        The closed side of the interval.
        If None, defaults to the value of ``label`` if provided, otherwise defaults to 'left'.

    Returns
    -------
    DataArray
        An variable with shape ``(n, 2)`` representing the bounds.

    Raises
    ------
    ValueError
        If the variable is not 1D.
    TypeError
        If the variable does not have compatible index.
    ValueError
        If a regular frequency cannot be inferred from the index.
    """
    try:
        index = obj.to_index()
    except ValueError:
        raise ValueError(
            'bounds inference is only supported for 1-dimensional coordinates.'
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
    """Convert a :py:class:`pandas.IntervalIndex` into a bounds variable.

    Parameters
    ----------
    interval : pd.IntervalIndex
        An interval index to convert.
    label : Literal['left', 'middle', 'right'], optional
        Which side or midpoint of the interval the index coordinate should label.
        If None, defaults to the closed side of the interval.
    dim : Hashable, optional
        The dimension name for the axis coordinate.
        If None, defaults to the name of the interval index.
    name : Hashable, optional
        The variable name for the axis coordinate. Defaults to the computed value of ``dim``.
        This is only needed if you want the variable name to be different from the dimension name.

    Returns
    -------
    DataArray
        A variable with shape ``(n, 2)``.

    Raises
    ------
    ValueError
        If the interval has an unsupported closed attribute.
    ValueError
        If the interval has no name and no dimension name is provided.
    """
    if interval.closed not in ClosedSide:
        raise ValueError(
            f'IntervalIndex has unsupported closed attribute: {interval.closed}. '
            f'Must be one of {list(ClosedSide)}.'
        )

    dim = dim or interval.name

    if dim is None:
        raise ValueError(
            'if the interval has no name, a dimension name must be provided.'
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
    """Convert a 2D variable to a :py:class:`pandas.IntervalIndex`.

    Parameters
    ----------
    obj : DataArray
        A 2-dimensional variable with size 2 along the second dimension.

    Returns
    -------
    pd.IntervalIndex
        A pandas interval index representing the bounds.
    """
    if obj.ndim != 2:
        raise ValueError(f'bounds must be a 2D DataArray, got {obj.ndim}D.')

    if obj.sizes.get(obj.dims[1], 0) != 2:
        raise ValueError(
            f'second dimension of bounds must have size 2, got size {obj.sizes.get(obj.dims[1], 0)}.'
        )

    closed = ClosedSide(obj.attrs.get('closed', 'left'))
    dim = obj.dims[0]

    return pd.IntervalIndex.from_arrays(
        *obj.transpose(),
        closed=closed.value,
        name=dim,
    )
