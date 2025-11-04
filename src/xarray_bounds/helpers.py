from __future__ import annotations

from collections.abc import Hashable
from typing import Literal, assert_never, cast

import numpy as np
import pandas as pd
import xarray as xr
from attr import dataclass

from xarray_bounds.types import (
    ClosedSide,
    LabelSide,
    is_date_offset,
    is_datetime_index,
)

__all__ = [
    'datetime_to_interval',
    'index_to_interval',
    'infer_midpoint_freq',
    'resolve_dim_name',
]


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
        The default is ‘left’ for all frequencies except for ‘ME’, ‘YE’,
        ‘QE’, and ‘W’ which have a default of ‘right’.
    closed : Literal['left', 'right'], optional
        Which side of bin interval is closed.
        The default is ‘left’ for all frequencies except for ‘ME’, ‘YE’,
        ‘QE’, and ‘W’ which have a default of ‘right’.
    name : Hashable, optional
        The name of the interval index. If None, the name of the index is used.
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

    See Also
    --------
    infer_freq_from_midpoint : Infer the frequency of an index of time midpoints.
    """
    if not is_datetime_index(index):
        raise TypeError(
            f'Expected a datetime-like index, got {type(index)=!r}'
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
        closed=closed,  # type: ignore[arg-type]
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
        If the index is not monotonic increasing
    ValueError
        If the index is not uniformly spaced
    """
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
        case _:
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
    closed: Literal['left', 'right'], default 'left'
        Whether to align the inferred frequency to the start or the end of the
        period.

    Returns
    -------
    str | None
        The inferred frequency string or None if the frequency cannot be inferred.

    Raises
    ------
    TypeError
        If the index is not datetime-like.
    ValueError
        If the index contains too few elements to infer a frequency or if the
        index is not 1D.

    See Also
    --------
    infer_freq : Infer the frequency of a time index
    """
    if not is_datetime_index(index):
        raise TypeError(
            f'Expected a datetime-like index, got {type(index)=!r}'
        )

    if index.size < 4:
        # Pandas needs >= 3 elements, but we lose 1 when calculating the diffs
        raise ValueError(
            'An index must have at least 4 values to infer frequency '
            f'using the midpoint method; got {index.size=!r}'
        )

    if not index.is_monotonic_increasing:
        raise ValueError('Index must be monotonic increasing.')

    if pd.infer_freq(index) == 'D':  # pragma: no cover
        # Shortcut for daily index
        return 'D'

    delta = pd.to_timedelta(index.diff())  # pyright: ignore [reportAttributeAccessIssue]
    left = (index - delta / 2).dropna()

    if (freq := pd.infer_freq(left)) and freq != 'h':
        # Avoid spurious hourly frequencies
        return freq
    # The remaining possibilities are 'MS', 'ME', 'QS', 'QE', 'YS', 'YE' (with
    # anchors), which are all 'multiples' of monthly frequencies, so we can
    # snap to a monthly frequency and have pandas attempt to infer the frequency.
    # Snapping will also conveniently handle the n-1 problem produced by diffing.
    freq = 'ME' if closed == 'right' else 'MS'
    return pd.infer_freq(left.snap(freq).normalize())


def resolve_dim_name(obj: xr.Dataset | xr.DataArray, key: str) -> str:
    """Resolve a dimension name from an axis key.

    The key can be a dimension name or any key understood by ``cf-xarray``.

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        The xarray object to get the dimension name from
    key : str
        The dimension name or CF axis key

    Returns
    -------
    str
        The dimension name

    Raises
    ------
    KeyError
        If no dimension is found for the given axis.
    """
    if key in obj.dims:
        return key
    try:
        # cf-xarray will raise if the key is not found...
        dim = obj.cf[key].name
        # but it might find a variable that is not a dimension.
        if dim not in obj.dims:
            raise KeyError
    except KeyError:
        raise KeyError(f'No dimension found for key {key!r}.')
    return dim


@dataclass(frozen=True)
class OffsetAlias:
    """An object representing a Pandas offset alias: a ``freq`` string.

    This class is not normally instantiated directly. Instead, use
    `OffsetAlias.from_freq` to parse a frequency string or `pd.DateOffset`
    object.

    Attributes
    ----------
    base : str
        The base frequency alias.
    n : int, optional
        The multiplier for the base frequency, default is 1.
    alignment : Literal['S', 'E'] | None, optional
        The alignment for the frequency, default is None.
    anchor : str | None, optional
        The anchor for the frequency, default is None.
    """

    n: int
    base: str
    alignment: Literal['S', 'E'] | None = None
    anchor: str | None = None

    def __str__(self) -> str:
        value = self.base
        if self.n < 0 or self.n > 1:
            value = f'{self.n}{value}'
        if self.alignment:
            value = f'{value}{self.alignment}'
        if self.anchor:
            value = f'{value}-{self.anchor}'
        return value

    @property
    def is_end_aligned(self) -> bool:
        """Whether the offset alias is end-aligned."""
        return self.base == 'W' or self.alignment == 'E'

    @classmethod
    def from_freq(cls, freq: str | pd.DateOffset) -> OffsetAlias:
        """Parse a Pandas offset alias into its components."""
        return _parse_freq(freq)

    @property
    def freqstr(self):
        """String representation of the offset alias."""
        return str(self)

    def to_offset(self):
        """Corresponding pandas DateOffset object."""
        return pd.tseries.frequencies.to_offset(str(self))


def _parse_freq(
    freq: str | pd.DateOffset, is_period: bool = False
) -> OffsetAlias:
    """Parse a Pandas frequency string into its components.

    Parameters
    ----------
    freq : str
        The frequency string to parse
    is_period : bool, default False
        Whether the frequency is for a period.

    Returns
    -------
    OffsetAlias
        A ParsedFreq object

    Raises
    ------
    ValueError
        If the freq is invalid

    See Also
    --------
    https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases
    """
    offset = pd.tseries.frequencies.to_offset(freq, is_period=is_period)
    alignment = None
    base, *rest = offset.name.split(sep='-')
    if base.endswith(('S', 'E')):
        alignment = cast(Literal['S', 'E'], base[-1])
        base = base.removesuffix(alignment)
    if is_period:
        alignment = None
    anchor = rest[0] if rest else None
    return OffsetAlias(
        n=offset.n,
        base=base,
        alignment=alignment,
        anchor=anchor,
    )


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


# def interval_to_bounds(
#     index: pd.IntervalIndex,
#     *,
#     label: LabelSide = LabelSide.LEFT,
# ) -> xr.DataArray:
#     """Convert an interval index to a bounds DataArray.
#
#     The name of the index is used as the dimension name.
#
#     Parameters
#     ----------
#     index : pd.IntervalIndex
#         The interval index to convert.
#     label : Literal['left', 'middle', 'right'], default 'left'
#         How to label the bounds.
#
#     Returns
#     -------
#     xr.DataArray
#         The bounds DataArray.
#     """
#     if not is_interval_index(index):
#         raise TypeError('"index" must be an IntervalIndex.')
#
#     dim = index.name
#     name = f'{dim}_{OPTIONS["bounds_dim"]}'
#     data = np.stack(arrays=(index.left, index.right), axis=1)
#     labels = getattr(index, 'mid' if label == LabelSide.MIDDLE else label)
#
#     return xr.DataArray(
#         name=name,
#         data=data,
#         coords={dim: (dim, labels)},
#         dims=(dim, OPTIONS['bounds_dim']),
#         attrs={'closed': index.closed},
#     )
