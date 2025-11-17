"""Utility functions for inferring interval indices from pandas indices.

These function are publicly exposed, but it is recommended to use the higher-level
functions in :py:mod:`xarray_bounds.core` instead.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import assert_never

import numpy as np
import pandas as pd

from xarray_bounds.helpers import OffsetAlias
from xarray_bounds.types import (
    ClosedSide,
    LabelSide,
    is_date_offset,
    is_datetime_index,
    is_multi_index,
)

__all__ = [
    'datetime_to_interval',
    'index_to_interval',
    'infer_midpoint_freq',
]


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
    if is_datetime_index(index):
        return datetime_to_interval(
            index=index, closed=closed, label=label, name=name
        )

    return index_to_interval(
        index=index, label=label, closed=closed, name=name
    )


def datetime_to_interval(
    index: pd.DatetimeIndex,
    *,
    label: LabelSide | None = None,
    closed: ClosedSide | None = None,
    name: Hashable | None = None,
    normalize: bool = False,
) -> pd.IntervalIndex:
    """Return an interval index from a datetime index with a regular frequency.

    This function is useful when the labels of a datetime index represent
    intervals--rather than instants--in time. For example, the index may
    represent a time axis for consecutive months whose values contain monthly means.

    The intervals can be thought of as the bins that would be used to create
    the input index when using a pandas/xarray `resample` operation with the
    provided `label` and `closed` arguments.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The datetime index to infer bounds for.
    label : Literal['left', 'middle', 'right'], optional
        Which bin edge or midpoint the index labels.
        The default is `left` for all frequencies except for `ME`, `YE`,
        `QE`, and `W` which have a default of `right`.
    closed : Literal['left', 'right'], optional
        Which side of bin interval is closed.
        The default is `left` for all frequencies except for `ME`, `YE`,
        `QE`, and `W` which have a default of `right`.
    name : Hashable, optional
        The name of the interval index. If None, the name of the index is used.
    normalize : bool, default False
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
        If the index has a timezone-aware offset.

    See Also
    --------
    infer_freq_from_midpoint : Infer the frequency of an index of time midpoints.
    """
    if not is_datetime_index(index):
        raise TypeError(
            f'expected a {pd.DatetimeIndex!r}, got {type(index)!r}'
        )
    try:
        if label == LabelSide.MIDDLE:
            freq = infer_midpoint_freq(index, closed)
        else:
            freq = pd.infer_freq(index)
        if freq is None:
            raise ValueError
    except ValueError as error:
        raise ValueError('Could not infer a regular frequency.') from error

    offset = pd.tseries.frequencies.to_offset(freq)
    assert is_date_offset(offset)

    if index.tz is not None and offset.n == 1:
        raise ValueError(
            '"datetime_to_interval" does not currently handle DST-aware offsets safely. '
            'Consider converting "index" to UTC.'
        )

    descending = index.is_monotonic_decreasing

    alias = OffsetAlias.from_freq(freq)
    if label is None:
        label = LabelSide.RIGHT if alias.is_end_aligned else LabelSide.LEFT

    if closed is None:
        closed = ClosedSide.RIGHT if alias.is_end_aligned else ClosedSide.LEFT

    match label:
        case LabelSide.LEFT:
            left = index
        case LabelSide.MIDDLE:
            left = _left_from_midpoint(index, offset)
        case LabelSide.RIGHT:
            left = index - offset
        case _:
            assert_never(label)

    right = left + offset

    if normalize:
        left = left.normalize()
        right = right.normalize()

    if descending:
        left, right = right, left
        closed = (
            ClosedSide.RIGHT if closed == ClosedSide.LEFT else ClosedSide.LEFT
        )

    return pd.IntervalIndex.from_arrays(
        left=left,
        right=right,
        closed=str(closed),  # type: ignore[arg-type]
        name=name or index.name,
    )


def index_to_interval(
    index: pd.Index,
    label: LabelSide = LabelSide.LEFT,
    closed: ClosedSide = ClosedSide.LEFT,
    name: Hashable | None = None,
) -> pd.IntervalIndex:
    """Return an interval index representing the bounds of an index.

    Parameters
    ----------
    index : pd.Index
        The index to infer bounds for.
    label : Literal['left', 'middle', 'right'], optional
        Which bin edge or midpoint the index labels.
    closed : Literal['left', 'right'], optional
        Which side of the bin interval is closed.
    name : Hashable, optional
        The name of the interval index. If None, the name of the index is used.

    Returns
    -------
    pd.IntervalIndex
        The interval index representing the bounds.

    Raises
    ------
    ValueError
        If the index is too short
    ValueError
        If the index is not monotonic
    ValueError
        If the index is not uniformly spaced
    """
    if is_multi_index(index):
        raise ValueError('MultiIndex is not supported.')

    if len(index) < 2:
        raise ValueError(
            f'Index must have at least two elements, got {len(index)}'
        )

    descending = index.is_monotonic_decreasing
    if not index.is_monotonic_increasing and not descending:
        raise ValueError('Index must be monotonic.')

    diffs = np.diff(index)
    if not np.allclose(diffs, diffs[0]):
        raise ValueError(
            'Index is not uniformly spaced; cannot infer consistent intervals.'
        )
    step = diffs[0]

    match label:
        case LabelSide.LEFT:
            breaks = np.append(arr=index, values=index[-1] + step)
        case LabelSide.RIGHT:
            breaks = np.insert(arr=index, obj=0, values=index[0] - step)
        case LabelSide.MIDDLE:
            breaks = np.concatenate(
                [[index[0] - step / 2], index.to_numpy() + step / 2]
            )
        case _:  # pragma: no cover
            assert_never(label)

    if descending:
        breaks = breaks[::-1]
        closed = (
            ClosedSide.RIGHT if closed == ClosedSide.LEFT else ClosedSide.LEFT
        )

    interval = pd.IntervalIndex.from_breaks(
        breaks=breaks,
        closed=closed.value,
        name=name or index.name,
    )

    if descending:
        return interval[::-1]
    return interval


def infer_midpoint_freq(
    index: pd.DatetimeIndex,
    closed: ClosedSide | None = None,
) -> str | None:
    """Infer the frequency of a datetime index by comparing the differences
    between consecutive elements.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The index to infer the frequency from.
    closed: Literal['left', 'right'], optional
        Whether to align the inferred frequency to the start or the end of the
        period.

    Returns
    -------
    str | None
        The inferred frequency string or None if the frequency cannot be inferred.

    Raises
    ------
    TypeError
        If the index is not a datetime index.
    ValueError
        If the index contains too few labels to infer a frequency or if the
        index is not 1D.

    See Also
    --------`
    infer_freq : Infer the frequency of a time index
    """
    if not is_datetime_index(index):
        raise TypeError(
            f'expected a {pd.DatetimeIndex!r}, got {type(index)!r}'
        )

    if index.size < 4:
        # Pandas needs >= 3 values, but we lose 1 when calculating the diffs
        raise ValueError(f'index must have size >= 4; got {index.size=!r}')

    if not (index.is_monotonic_increasing or index.is_monotonic_decreasing):
        raise ValueError('index must be monotonic.')

    if pd.infer_freq(index) == 'D':
        # Shortcut for daily frequencies
        return 'D'

    delta = pd.to_timedelta(index.diff())  # type: ignore[attr-defined]
    left = (index - delta / 2).dropna()

    if freq := pd.infer_freq(left):
        # Avoid inferring exotic frequencies
        alias = OffsetAlias.from_freq(freq)
        if alias.base in {'D', 'W', 'M', 'Q', 'Y'}:
            return freq

    # The remaining possibilities are  all 'multiples' of monthly frequencies,
    # so we can snap to a monthly frequency and have pandas attempt to infer
    # the frequency. Snapping will also conveniently handle the n-1 problem
    # produced by diffing.
    freq = 'ME' if closed == 'right' else 'MS'
    return pd.infer_freq(left.snap(freq).normalize())


def _left_from_midpoint(
    index: pd.DatetimeIndex, offset: pd.DateOffset
) -> pd.DatetimeIndex:
    """Compute the left edges of intervals for a midpoint-labeled datetime index.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The midpoint index.
    offset : pd.DateOffset
        The frequency of the intervals.

    Returns
    -------
    pd.DatetimeIndex
        The left edges of each interval.
    """
    delta = pd.to_timedelta(index.diff())  # type: ignore[attr-defined]
    left = (index - delta / 2).dropna()
    aligned = left.shift(-1, freq=offset).normalize()
    if not aligned.freq == offset:
        print(Warning('could not align midpoints to the specified frequency.'))
    return pd.to_datetime(aligned.union([aligned[-1] + offset]))
