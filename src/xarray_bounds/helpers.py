from typing import Literal, cast
import warnings
from attr import dataclass
import pandas as pd
import numpy as np
import xarray as xr

from xarray_bounds.types import (
    IntervalClosed,
    IntervalLabel,
    is_date_offset,
    is_datetime_index,
    LabelSide,
    ClosedSide,
)
from xarray_bounds.options import OPTIONS

__all__ = [
    'datetime_to_interval',
    'index_to_interval',
    'infer_bounds',
    'infer_interval',
]


@dataclass(frozen=True)
class ParsedFreq:
    """A parsed pandas frequency string.

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

    @property
    def freqstr(self) -> str:
        """Return the frequency string."""
        freqstr = self.base
        if self.n > 1:
            freqstr = f'{self.n}{freqstr}'
        if self.alignment:
            freqstr = f'{freqstr}{self.alignment}'
        if self.anchor:
            freqstr = f'{freqstr}-{self.anchor}'
        return freqstr


def parse_freq(freq: str, is_period: bool = False) -> ParsedFreq:
    """Parse a Pandas frequency string into its components.

    Parameters
    ----------
    freq : str
        The frequency string to parse
    is_period : bool, default False
        Whether the frequency is for a period.

    Returns
    -------
    ParsedFreq
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
    return ParsedFreq(
        n=offset.n,
        base=base,
        alignment=alignment,
        anchor=anchor,
    )


def infer_midpoint_freq(
    obj: xr.DataArray | pd.DatetimeIndex,
    how: Literal['start', 'end'] = 'start',
) -> str | None:
    """
    Infer the frequency of a datetime index by comparing the differences
    between consecutive elements.

    Parameters
    ----------
    obj : xr.DataArray | pd.DatetimeIndex
        The object to infer the frequency from.
    how: Literal['start', 'end'], default 'start'
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
    if isinstance(obj, xr.DataArray):
        index = obj.cf['time'].to_index()
    else:
        index = obj
    assert isinstance(index, pd.DatetimeIndex)

    if index.size < 4:
        # Pandas needs >= 3 elements, but we lose 1 when calculating the diffs
        raise ValueError(
            'An index must have at least 4 values to infer frequency '
            f'using the midpoint method; got {index.size=!r}'
        )

    if pd.infer_freq(index) == 'D':  # pragma: no cover
        return 'D'

    delta = index.diff()  # pyright: ignore [reportAttributeAccessIssue]
    assert isinstance(delta, pd.TimedeltaIndex)
    left = (index - delta / 2).dropna()

    if freq := pd.infer_freq(left):
        if parse_freq(freq).base in ['D', 'W', 'M', 'Q', 'Y']:
            return freq
    # The remaining possibilities are 'MS', 'ME', 'QS', 'QE', 'YS', 'YE',
    # which are all 'multiples' of monthly frequencies, so we can snap to the
    # monthly frequency and infer the frequency from there.
    snap_freq = 'MS' if how == 'start' else 'ME'
    snapped = left.snap(snap_freq).normalize()
    return pd.infer_freq(snapped)


def infer_freq(
    obj: xr.DataArray | pd.DatetimeIndex,
) -> str | None:
    """Return the most likely frequency of a pandas or xarray data object.

    This method first tries to infer the frequency using xarray's `infer_freq`
    method. If that fails, or if the inferred frequency is sub-daily, it falls
    back to the `infer_midpoint_freq` method.

    Parameters
    ----------
    index : pd.DatetimeIndex | xr.DataArray
        The index to infer the frequency from. If passed a Series or a
        DataArray, it will use the values of the object NOT the index.

    Returns
    -------
    str | None
        The frequency string or None if the frequency cannot be inferred.

    Raises
    ------
    TypeError
        If the index is not datetime-like.
    ValueError
        If the index has too few elements to infer a frequency or the index
        is not 1D.

    See Also
    --------
    infer_midpoint_freq : Infer the frequency of a time index of midpoints.
    """
    freq = xr.infer_freq(obj)
    if freq and freq.endswith('h'):
        # Some midpoint frequencies are inferred as 'h' (hourly) frequencies,
        # we consider this a failure.
        freq = None
    if freq is None:
        freq = infer_midpoint_freq(obj)
    return freq


def datetime_to_interval(
    index: pd.DatetimeIndex,
    *,
    label: IntervalLabel | None = None,
    closed: IntervalClosed | None = None,
    normalize: bool = True,
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
    infer_midpoint_freq : Infer the frequency of an index of time midpoints.

    Notes
    -----
    Passing `'middle'` to the `label` parameter will coerce the index
    to its inferred frequency (see `infer_midpoint_freq` for details) and is
    equivalent to passing `label = None`.
    """
    try:
        freq = infer_freq(index)
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
        index = pd.to_datetime(
            index.to_series().asfreq(freq).index
        ).normalize()
        assert is_date_offset(index.freq)
        index = pd.to_datetime(index.union([index[-1] + index.freq])).shift(-1)

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
    """Infer bounds for a 1D coordinate.

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
    interval = infer_interval(index=index, closed=closed, label=label)
    data = np.stack(arrays=(interval.left, interval.right), axis=1)
    name = f'{dim}_{OPTIONS["bounds_name"]}'
    coord = obj.assign_attrs(bounds=name)
    return xr.DataArray(
        name=name,
        data=data,
        coords={dim: coord},
        dims=(dim, OPTIONS['bounds_name']),
        attrs={'closed': interval.closed},
    )


def infer_interval(
    index: pd.Index,
    *,
    closed: IntervalClosed | None = None,
    label: IntervalLabel | None = None,
) -> pd.IntervalIndex:
    """Infer an interval index from a pandas index.

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
            # Try first to infer bounds from frequency
            interval = datetime_to_interval(
                index=index, closed=closed, label=label
            )
        except ValueError:
            # Fallback to midpoint inference for irregular datetime indexes
            warnings.warn(
                'Failed to infer bounds from datetime index frequency, '
                'falling back to midpoint inference.'
            )
            return midpoint_to_interval(index=index, closed=closed)
        else:
            return interval
    return index_to_interval(index=index, label=label, closed=closed)


def index_to_interval(
    index: pd.Index,
    label: IntervalLabel | None = None,
    closed: IntervalClosed | None = None,
) -> pd.IntervalIndex:
    """Return an interval index representing the bounds of an index.

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
        return midpoint_to_interval(index=index, closed=closed)
    label = label or 'left'
    closed = closed or label
    step = np.diff(index).mean()
    if label == 'left':
        index = index.union([index[-1] + step])
    else:
        index = index.union([index[0] - step])
    return pd.IntervalIndex.from_breaks(breaks=index, closed=closed)


def midpoint_to_interval(
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
