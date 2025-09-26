from typing import cast
import warnings
import pandas as pd
import numpy as np
import xarray as xr

from xarray_bounds.types import (
    IntervalClosed,
    IntervalLabel,
    is_datetime_index,
    LabelSide,
    ClosedSide,
)

__all__ = ['datetime_to_interval', 'infer_bounds']


def datetime_to_interval(
    index: pd.DatetimeIndex,
    *,
    label: IntervalLabel | None = None,
    closed: IntervalClosed | None = None,
    normalize: bool = True,
) -> pd.IntervalIndex:
    """
    Return an interval index representing the vertices of the boundaries of
    each datetime in the index.

    This function is useful when a datetime index does not represent point
    values, but rather intervals of time. For example, the index may represent
    a time axis for consecutive months whose values contain monthly means.

    The intervals can be thought of as the bins that would be used to create
    the input index when using a pandas/xarray `resample` operation with the
    provided `label` and `closed` arguments.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The datetime index to infer bounds for.
    label : Literal['left', 'middle', 'right'], optional
        Which bin edge or midpoint the index labels.
        The default is ‘left’ for all frequency offsets except for ‘ME’, ‘YE’,
        ‘QE’, and ‘W’ which all have a default of ‘right’.
    closed : Literal['left', 'right'], optional
        Which side of bin interval is closed.
        The default is ‘left’ for all frequency offsets except for ‘ME’, ‘YE’,
        ‘QE’, and ‘W’ which all have a default of ‘right’.
    normalize : bool, default True
        If True, the bounds will be normalized to midnight.

    Returns
    -------
    pd.IntervalIndex
        The interval index representing the bounds.

    Raises
    ------
    TypeError
        If the index is not a datetime index.
    ValueError
        If a regular frequency cannot be inferred from the index.
    ValueError
        If the passed frequency does not conform to the index frequency.

    See Also
    --------
    infer_midpoint_freq : Infer the frequency of an index of time midpoints.

    Notes
    -----
    Passing `'middle'` to the `label` parameter will coerce the index
    to its inferred frequency (see `infer_midpoint_freq` for details) and is
    equivalent to passing `label = None`.
    """
    try:
        freq = pd.infer_freq(index)
        assert freq is not None
    except (ValueError, AssertionError) as error:
        raise ValueError(
            'To convert a datetime index to an interval index, '
            'the index must have a regular frequency.'
        ) from error

    resolved_label = _resolve_label(label, freq)
    resolved_closed = _resolve_closed(closed, freq)

    if resolved_label == LabelSide.MIDDLE:
        # Coerce the midpoints to their inferred frequency
        index = cast(
            pd.DatetimeIndex, index.to_series().asfreq(freq).index
        ).normalize()
        _offset = index.freq
        assert isinstance(_offset, pd.tseries.offsets.BaseOffset)
        index = pd.to_datetime(index.union([index[-1] + _offset])).shift(-1)

    if resolved_label == LabelSide.RIGHT:
        left = index.shift(periods=-1, freq=freq)
    else:
        left = index

    right = left.shift(periods=1, freq=freq)
    if normalize:
        left = left.normalize()  # pyright: ignore [reportAttributeAccessIssue]
        right = right.normalize()  # pyright: ignore [reportAttributeAccessIssue]

    return pd.IntervalIndex.from_arrays(
        left=left,
        right=right,
        closed=resolved_closed,
    )


def infer_bounds(
    obj: xr.DataArray,
    *,
    closed: IntervalClosed | None = None,
    label: IntervalLabel | None = None,
) -> xr.DataArray:
    """Infer bounds for an 1D coordinate.

    Parameters
    ----------
    obj : DataArray
        The data array to infer bounds from.
    closed : Literal['left', 'right'], optional
        The closed side of the interval.
    label : Literal['left', 'middle', 'right'], optional
        Which bin edge or midpoint the index labels.

    Returns
    -------
    DataArray
        The bounds data array.

    Raises
    ------
    ValueError
        If the dimension is not indexed.
    ValueError
        If the data array is not 1D.
    """
    if obj.ndim != 1:
        raise ValueError(
            'Bounds are currently only supported for 1D coordinates.'
        )
    dim = obj.dims[0]
    index = obj.to_index()
    interval = _infer_interval(index=index, closed=closed, label=label)
    data = np.stack(arrays=(interval.left, interval.right), axis=1)
    name = f'{dim}_bounds'
    coord = obj.assign_attrs(bounds=name)

    return xr.DataArray(
        name=name,
        data=data,
        coords={dim: coord},
        dims=(dim, 'bounds'),
        attrs={'closed': interval.closed},
    )


def _infer_interval(
    index: pd.Index,
    *,
    closed: IntervalClosed | None = None,
    label: IntervalLabel | None = None,
) -> pd.IntervalIndex:
    """Infer an interval index from a Pandas index.

    If the index is a datetime index with a regular frequency, the bounds are
    inferred from the frequency. Otherwise, the bounds are inferred assuming
    the index represents the midpoints of the intervals.

    Parameters
    ----------
    index : pd.Index
        The index to infer bounds for.
    closed : Literal['left', 'right'] | None, optional
        The closed side of the interval.
    label : Literal['left', 'middle', 'right'] | None, optional
        Which bin edge or midpoint the index labels.

    Returns
    -------
    pd.IntervalIndex
        The interval index representing the bounds.
    """
    if is_datetime_index(index):
        try:
            # First try to infer bounds from frequency
            interval = datetime_to_interval(
                index=index, closed=closed, label=label
            )
        except ValueError:
            # Fallback to midpoint inference for irregular datetime indexes
            warnings.warn(
                'Failed to infer bounds from datetime index frequency, '
                'falling back to midpoint inference.'
            )
            return _midpoint_to_interval(index=index, closed=closed)
        else:
            return interval
    return _index_to_interval(index=index, label=label, closed=closed)


def _index_to_interval(
    index: pd.Index,
    label: IntervalLabel | None = None,
    closed: IntervalClosed | None = None,
) -> pd.IntervalIndex:
    """
    Return an interval index representing the bounds of an index.

    Parameters
    ----------
    index : pd.Index
        The index to infer bounds for.
    label : Literal['left', 'middle', 'right'], optional
        Which bin edge or midpoint the index labels.
    closed : Literal['left', 'right'], optional
        Which side of bin interval is closed.

    Returns
    -------
    pd.IntervalIndex
        The interval index representing the bounds.
    """
    if label == 'middle':
        return _midpoint_to_interval(index=index, closed=closed)
    label = label or 'left'
    closed = closed or label
    step = np.diff(index).mean()
    if label == 'left':
        index = index.union([index[-1] + step])
    else:
        index = index.union([index[0] - step])
    return pd.IntervalIndex.from_breaks(breaks=index, closed=closed)


def _midpoint_to_interval(
    index: pd.Index | np.ndarray,
    closed: IntervalClosed | None = None,
) -> pd.IntervalIndex:
    """
    Return an interval index representing the bounds of an index of midpoints.

    Parameters
    ----------
    index : pd.Index | np.ndarray
        The index of midpoints to infer bounds for.
    closed : Literal['left', 'right'], optional
        Which side of bin interval is closed. Defaults to 'left'.

    Returns
    -------
    pd.IntervalIndex
        The interval index representing the bounds.
    """
    closed = closed or 'left'
    diffs = np.diff(index)
    # Assume that the first difference is the same as the second...
    diffs = np.insert(arr=diffs, obj=0, values=diffs[0])
    left = index - diffs / 2
    right = index + diffs / 2
    return pd.IntervalIndex.from_arrays(left=left, right=right, closed=closed)


def _is_end_aligned(freq: str) -> bool:
    if freq == 'W':
        return True
    offset = pd.tseries.frequencies.to_offset(freq)
    return type(offset).__name__.lower().endswith('end')


def _resolve_label(label: IntervalLabel | None, freq: str) -> IntervalLabel:
    if label is not None:
        return LabelSide(label).value
    return (
        LabelSide.RIGHT.value
        if _is_end_aligned(freq)
        else LabelSide.LEFT.value
    )


def _resolve_closed(
    closed: IntervalClosed | None, freq: str
) -> IntervalClosed:
    if closed is not None:
        return ClosedSide(closed).value
    return (
        ClosedSide.RIGHT.value
        if _is_end_aligned(freq)
        else ClosedSide.LEFT.value
    )
