"""Utility functions for inferring interval indices from pandas indices.

These function are publicly exposed, but it is recommended to use the higher-level
functions in :py:mod:`xarray_bounds.core`.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable
from datetime import UTC
from typing import assert_never

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from xarray_bounds._helpers import OffsetAlias
from xarray_bounds.constants import (
    IntervalClosed,
    IntervalLabel,
)

__all__ = [
    "datetime_to_interval",
    "index_to_interval",
    "infer_interval",
    "infer_midpoint_freq",
]


def _flip_label(label: IntervalLabel | None) -> IntervalLabel | None:
    """Return the label for the opposite side of the interval.

    ``None`` and ``IntervalLabel.MIDDLE`` are returned unchanged.
    """
    if label is None or label is IntervalLabel.MIDDLE:
        return label
    return (
        IntervalLabel.RIGHT
        if label is IntervalLabel.LEFT
        else IntervalLabel.LEFT
    )


def _flip_closed(closed: IntervalClosed | None) -> IntervalClosed | None:
    """Return the closed side for the opposite side of the interval.

    ``None`` is returned unchanged.
    """
    if closed is None:
        return closed
    return (
        IntervalClosed.RIGHT
        if closed is IntervalClosed.LEFT
        else IntervalClosed.LEFT
    )


def infer_interval(
    index: pd.Index,
    *,
    label: IntervalLabel | None = None,
    closed: IntervalClosed | None = None,
    name: Hashable | None = None,
    normalize: bool = False,
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
        If None, a suitable default is chosen by the underlying implementation.
    closed : Literal['left', 'right'] | None, optional
        The closed side of the interval.
        If None, a suitable default is chosen by the underlying implementation.
    name : Hashable | None, optional
        The name of the interval index. If None, the name of the index is used.
    normalize : bool, optional
        Whether to normalize the datetime index. Ignored for other index types.

    Returns
    -------
    pd.IntervalIndex
        The interval index representing the bounds.
    """
    if isinstance(index, pd.DatetimeIndex):
        return datetime_to_interval(
            index=index,
            closed=closed,
            label=label,
            name=name,
            normalize=normalize,
        )

    return index_to_interval(
        index=index,
        label=label,
        closed=closed,
        name=name,
    )


def datetime_to_interval(
    index: pd.DatetimeIndex,
    *,
    label: IntervalLabel | None = None,
    closed: IntervalClosed | None = None,
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

    The index may be monotonic increasing or decreasing. Decreasing indices
    are handled by mirroring the increasing case: the ``label`` and ``closed``
    arguments are interpreted relative to the direction of the axis, and the
    resulting intervals are returned in the original (decreasing) order.

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
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(
            f"expected a {pd.DatetimeIndex!r}, got {type(index)!r}"
        )

    label_side: IntervalLabel | None = (
        IntervalLabel(label) if label is not None else None
    )
    closed_side: IntervalClosed | None = (
        IntervalClosed(closed) if closed is not None else None
    )

    if index.is_monotonic_decreasing and not index.is_monotonic_increasing:
        # Mirror to the increasing case (the same convention as
        # ``index_to_interval``): reverse the index and the label/closed
        # sides, then reverse the result.
        # Note: empty and single-element indices are monotonic in both
        # directions, so they are excluded here.
        result = datetime_to_interval(
            index[::-1],
            label=_flip_label(label_side),
            closed=_flip_closed(closed_side),
            name=name,
            normalize=normalize,
        )
        return result[::-1]

    if label_side == IntervalLabel.MIDDLE:
        freq = infer_midpoint_freq(index, closed_side)
    else:
        freq = pd.infer_freq(index)

    if freq is None:
        raise ValueError("index must have an inferrable frequency.")

    offset = pd.tseries.frequencies.to_offset(freq)
    assert isinstance(offset, pd.DateOffset)

    if index.tz not in [None, UTC]:
        raise ValueError(
            f"expected 'index' with timezone 'UTC', got {str(index.tz)!r}"
        )

    alias = OffsetAlias.from_freq(offset)
    if label_side is None:
        label_side = (
            IntervalLabel.RIGHT if alias.is_end_aligned else IntervalLabel.LEFT
        )

    if closed_side is None:
        closed_side = (
            IntervalClosed.RIGHT
            if alias.is_end_aligned
            else IntervalClosed.LEFT
        )

    match label_side:
        case IntervalLabel.LEFT:
            left = index
        case IntervalLabel.MIDDLE:
            left = _left_from_midpoint(index, offset)
        case IntervalLabel.RIGHT:
            left = index - offset
        case _:
            assert_never(label_side)

    right = left + offset

    if normalize:
        left = left.normalize()
        right = right.normalize()

    return pd.IntervalIndex.from_arrays(
        left=left,
        right=right,
        closed=str(closed_side),  # type: ignore[arg-type]
        name=name or index.name,
    )


def index_to_interval(
    index: pd.Index,
    label: IntervalLabel | None = None,
    closed: IntervalClosed | None = None,
    name: Hashable | None = None,
) -> pd.IntervalIndex:
    """Return an interval index representing the bounds of an index.

    The index must have a numeric non-boolean dtype.

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
    TypeError
        If the index has a non-numeric or boolean dtype
    ValueError
        If the index is too short
    ValueError
        If the index is not monotonic
    ValueError
        If the index is not uniformly spaced
    """
    if not ptypes.is_numeric_dtype(index) or ptypes.is_bool_dtype(index):
        raise TypeError(
            f"expected {pd.Index!r} with non-boolean numeric dtype, got {index.dtype!r}"
        )
    if index.ndim != 1:
        raise ValueError("only 1-dimensional indices are supported.")

    if index.size < 2:
        raise ValueError(
            f'index must have at least two elements, got "{index.size}".'
        )

    if not (index.is_monotonic_increasing or index.is_monotonic_decreasing):
        raise ValueError("index must be monotonic.")

    diffs = np.diff(index)
    if not np.allclose(diffs, diffs[0]):
        raise ValueError("index must be uniformly spaced.")
    step = diffs[0]

    label_side = (
        IntervalLabel(label) if label is not None else IntervalLabel.LEFT
    )
    closed_side = (
        IntervalClosed(closed) if closed is not None else IntervalClosed.LEFT
    )

    match label_side:
        case IntervalLabel.LEFT:
            breaks = np.append(arr=index, values=index[-1] + step)
        case IntervalLabel.RIGHT:
            breaks = np.insert(arr=index, obj=0, values=index[0] - step)
        case IntervalLabel.MIDDLE:
            breaks = np.concatenate(
                [[index[0] - step / 2], index.to_numpy() + step / 2]
            )
        case _:  # pragma: no cover
            assert_never(label_side)

    if index.is_monotonic_decreasing:
        breaks = breaks[::-1]
        closed_side = (
            IntervalClosed.RIGHT
            if closed_side == IntervalClosed.LEFT
            else IntervalClosed.LEFT
        )

    interval = pd.IntervalIndex.from_breaks(
        breaks=breaks,
        closed=closed_side.value,
        name=name or index.name,
    )

    if index.is_monotonic_decreasing:
        return interval[::-1]
    return interval


def infer_midpoint_freq(
    index: pd.DatetimeIndex,
    closed: IntervalClosed | None = None,
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
    --------
    infer_freq : Infer the frequency of a time index
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(
            f"expected a {pd.DatetimeIndex!r}, got {type(index)!r}"
        )

    if index.size < 4:
        # Pandas needs >= 3 values, but we lose 1 when calculating the diffs
        raise ValueError(f"index must have size >= 4; got {index.size=!r}")

    if not (index.is_monotonic_increasing or index.is_monotonic_decreasing):
        raise ValueError("index must be monotonic.")

    if index.is_monotonic_decreasing and not index.is_monotonic_increasing:
        # Mirror to the increasing case (see ``datetime_to_interval``).
        return infer_midpoint_freq(index[::-1], closed=_flip_closed(closed))

    if pd.infer_freq(index) == "D":
        # Shortcut for daily frequencies
        return "D"

    delta = pd.to_timedelta(index.diff())  # type: ignore[attr-defined]
    left = (index - delta / 2).dropna()

    if freq := pd.infer_freq(left):
        # Avoid inferring exotic frequencies
        alias = OffsetAlias.from_freq(freq)
        if alias.base in {"D", "W", "M", "Q", "Y"}:
            return freq

    # The remaining possibilities are  all 'multiples' of monthly frequencies,
    # so we can snap to a monthly frequency and have pandas attempt to infer
    # the frequency. Snapping will also conveniently handle the n-1 problem
    # produced by diffing.
    freq = "ME" if closed == "right" else "MS"
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
    if aligned.freq != offset:
        warnings.warn(
            f"could not align midpoints with freq {aligned.freq!r} to the specified frequency ({offset!r})."
        )
    return pd.to_datetime(aligned.union([aligned[-1] + offset]))
