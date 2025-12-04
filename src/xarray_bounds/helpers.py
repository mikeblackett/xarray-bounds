"""Helper functions for xarray-bounds.

These functions are used internally by the package and are generally not intended
for public use.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any, Literal, cast

import pandas as pd
import xarray as xr
from attr import dataclass

from xarray_bounds.options import OPTIONS

__all__ = [
    'resolve_dim_name',
    'resolve_standard_name',
    'resolve_variable_name',
def validate_interval_label(
    label: str | None, default: str | None = None
) -> IntervalLabel:
    if label is not None:
        return LabelSide(label).value
    if label is None and default is None:
        return LabelSide.LEFT.value
    if label is None:
        return LabelSide(default).value


def validate_interval_closed(
    closed: str | None, default: str | None = None
) -> IntervalClosed:
    if closed is not None:
        return ClosedSide(closed).value
    if closed is None and default is None:
        return ClosedSide.LEFT.value
    if closed is None:
        return ClosedSide(default).value

    The key can be any value understood by ``cf-xarray``.

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


def resolve_bounds_name(dim: Hashable) -> str:
    """Get the standard name for bounds of a given dimension.

    Parameters
    ----------
    dim : Hashable
        The dimension name.

    Returns
    -------
    str
        The standard name for the bounds coordinate.
    """
    return f'{dim}_{OPTIONS["bounds_dim"]}'


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

    @property
    def offset(self):
        """Corresponding pandas DateOffset object."""
        return pd.tseries.frequencies.to_offset(str(self))


def _parse_freq(
    freq: str | pd.DateOffset, is_period: bool = False
) -> OffsetAlias:
    """Parse a Pandas frequency string into its components.

    Parameters
    ----------
    freq : str | pd.DateOffset
        The frequency to parse
    is_period : bool, default False
        Whether the frequency is for a period.

    Returns
    -------
    OffsetAlias
        A object representing the parsed frequency.

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


def mapping_or_kwargs[T](
    parg: Mapping[Any, T] | None,
    kwargs: Mapping[str, T],
    func_name: str,
) -> Mapping[Hashable, T]:
    """Return a mapping of arguments from either keyword arguments or a positional argument mapping."""
    if parg is None or parg == {}:
        return cast(Mapping[Hashable, T], kwargs)
    if kwargs:
        raise ValueError(
            f'cannot specify both keyword and positional arguments to {func_name}'
        )
    return parg
