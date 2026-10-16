from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd
import xarray as xr

from xarray_bounds._helpers import (
    validate_interval_closed,
    validate_interval_label,
)
from xarray_bounds.options import OPTIONS
from xarray_bounds.types import (
    ClosedSide,
    IntervalClosed,
    IntervalLabel,
    LabelSide,
    is_enum_member,
)
from xarray_bounds.utilities import infer_interval

__all__ = [
    "bounds_to_interval",
    "infer_bounds",
    "interval_to_bounds",
]


def infer_bounds(
    obj: xr.DataArray,
    *,
    label: IntervalLabel | None = None,
    closed: IntervalClosed | None = None,
) -> xr.DataArray:
    """Infer the bounds for a 1D DataArray.

    Parameters
    ----------
    obj : DataArray
        A 1-dimensional DataArray with a monotonic and regularly spaced index.
    label : Literal['left', 'middle', 'right'], optional
        Which side or midpoint of the interval the index labels.
        If None, defaults to the value of ``closed`` if provided, otherwise defaults to 'left'.
    closed : Literal['left', 'right'], optional
        The closed side of the interval.
        If None, defaults to the value of ``label`` if provided, otherwise defaults to 'left'.

    Returns
    -------
    DataArray
        A DataArray with shape ``(n, 2)`` representing the bounds.

    Raises
    ------
    ValueError
        If the DataArray is not 1D.
    TypeError
        If the DataArray does not have compatible index.
    ValueError
        If the DataArray's index is not regularly spaced.
    """
    try:
        index = obj.to_index()
    except ValueError as error:
        raise ValueError("index must be 1-dimensional") from error

    if index.name == OPTIONS["bounds_dim"]:
        raise ValueError(
            f'"obj.name" cannot be the same as the bounds dimension: {OPTIONS["bounds_dim"]}'
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

    bounds = f"{coord}_{OPTIONS['bounds_dim']}"
    obj.attrs["bounds"] = bounds

    return xr.DataArray(
        data=data,
        coords={coord: obj},
        dims=(dim, OPTIONS["bounds_dim"]),
        name=bounds,
        attrs={"closed": interval.closed},
    )


def interval_to_bounds(
    interval: pd.IntervalIndex,
    *,
    label: IntervalLabel | None = None,
    dim: Hashable | None = None,
    name: Hashable | None = None,
) -> xr.DataArray:
    """Return a bounds DataArray from a :py:class:`pandas.IntervalIndex`.

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
        A DataArray with shape ``(n, 2)`` representing the bounds.

    Raises
    ------
    ValueError
        If the interval has an unsupported closed attribute.
    ValueError
        If the interval/dimension name is the same as the bounds dimension name
    ValueError
        If the interval has no name and no dimension name is provided.
    """
    if not is_enum_member(ClosedSide, interval.closed):
        raise ValueError(
            f"IntervalIndex has unsupported closed attribute: {interval.closed}. "
            f"Must be one of {list(ClosedSide)}."
        )

    dim = dim or interval.name

    if dim is None:
        raise ValueError(
            "if the interval has no name, a dimension name must be provided."
        )

    if name is None:
        name = dim

    if name == OPTIONS["bounds_dim"]:
        raise ValueError(
            "dimension name conflict:"
            f" the dimension name can't be the same as the bounds dimension ({OPTIONS['bounds_dim']})."
        )

    label = validate_interval_label(label=label, default=interval.closed)

    bounds = f"{name}_{OPTIONS['bounds_dim']}"

    data = np.stack(arrays=(interval.left, interval.right), axis=1)
    axis = (
        pd.Index(
            getattr(interval, "mid" if label == "middle" else label),
            name=dim,
        )
        .to_series()
        .to_xarray()
        .assign_attrs(bounds=bounds)
        .cf.guess_coord_axis()
    )

    return xr.DataArray(
        data=data,
        dims=(dim, OPTIONS["bounds_dim"]),
        coords={name: axis},
        name=bounds,
        attrs={"closed": interval.closed},
    )


def bounds_to_interval(obj: xr.DataArray) -> pd.IntervalIndex:
    """Convert a 2D DataArray to a :py:class:`pandas.IntervalIndex`.

    Parameters
    ----------
    obj : DataArray
        A 2-dimensional DataArray with size 2 along the second dimension.

    Returns
    -------
    pd.IntervalIndex
        An interval index representing the bounds.

    Raises
    ------
    ValueError
        If the DataArray is not 2D.
    ValueError
        If the second dimension of the DataArray is not size 2.
    """
    if obj.ndim != 2:
        raise ValueError(
            f"obj must be 2-dimensional, got object with {obj.ndim!r} dimensions."
        )

    if (size := obj.sizes.get(obj.dims[1], 0)) != 2:
        raise ValueError(
            f"second dimension of obj must have size 2, got size {size}."
        )

    closed = ClosedSide(obj.attrs.get("closed", "left"))
    dim = obj.dims[0]

    return pd.IntervalIndex.from_arrays(
        *obj.transpose(),
        closed=closed.value,
        name=dim,
    )
