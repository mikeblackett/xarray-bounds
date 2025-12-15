"""Helper functions for xarray-bounds.

These functions are used internally by the package and are generally not intended
for public use.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast

import pandas as pd
import xarray as xr

from xarray_bounds.types import (
    ClosedSide,
    IntervalClosed,
    IntervalLabel,
    LabelSide,
)

__all__ = [
    'OffsetAlias',
    'resolve_standard_name',
    'resolve_variable_name',
    'mapping_or_kwargs',
]


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


def resolve_axis_name(obj: xr.Dataset | xr.DataArray, key: Hashable) -> str:
    """Return the valid Axis standard name for a given key.

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        An xarray object.
    key : Hashable
        The name of a variable.
        Can be any alias understood by :py:mod:`cf_xarray`.

    Returns
    -------
    str
        The valid Axis standard name

    Raises
    ------
    KeyError
        If no valid Axis standard name is found for the given key.
    """
    if key in obj.cf.axes:
        # The key is already an axis name
        return cast(str, key)

    # Key is either a variable name, standard name or invalid
    name = resolve_variable_name(obj=obj, key=key)

    # ``cf.axes`` is a mapping of axis name to variable names
    for axis, names in obj.cf.axes.items():
        if name in names:
            return axis

    raise KeyError(f'No valid CF axis found for key {key!r}.')


def resolve_standard_name(
    obj: xr.Dataset | xr.DataArray, key: Hashable
) -> str:
    """Return the standard name for a given key.

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        An xarray object.
    key : Hashable
        The name of a variable.
        Can be any alias understood by :py:mod:`cf_xarray`.

    Returns
    -------
    Hashable
        The standard name

    Raises
    ------
    KeyError
        If no standard name is found for the given key.
    """
    if key in obj.cf.standard_names:
        # The key is already a standard name
        return cast(str, key)

    # The key is a variable name, axis name, or invalid
    variable = resolve_variable_name(obj=obj, key=key)

    # ``cf.standard_names`` is a mapping of standard name to variable names
    for name in obj.cf.standard_names:
        if variable in obj.cf.standard_names[name]:
            return name

    raise KeyError(f'No valid CF coordinate found for key {key!r}.')


def resolve_variable_name(
    obj: xr.Dataset | xr.DataArray, key: Hashable
) -> Hashable:
    """Return the variable name for a given key.

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        An xarray object.
    key : Hashable
        The name of a variable.
        Can be any alias understood by :py:mod:`cf_xarray`.

    Returns
    -------
    Hashable
        The variable name

    Raises
    ------
    KeyError
        If no variable is found for the given key.
    """
    try:
        return obj.cf[key].name
    except KeyError:
        raise KeyError(f'No variable found for key {key!r}.')


@dataclass(frozen=True)
class OffsetAlias:
    """An object representing a parsed Pandas offset alias: a ``freq`` string.

    This class is not normally instantiated directly.
    Instead, use ``OffsetAlias.from_freq`` to parse a frequency string or ``pd.DateOffset`` object.

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
        A pandas frequency string or DateOffset object.
    is_period : bool, default False
        Whether the frequency is for a period.

    Returns
    -------
    OffsetAlias
        An object representing the parsed frequency.

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


T = TypeVar('T')


def mapping_or_kwargs(
    parg: Mapping[Any, T] | None,
    kwargs: Mapping[str, T],
    func_name: str,
) -> Mapping[Hashable, T]:
    """Return a mapping of arguments from either keyword arguments or a positional argument mapping.

    Parameters
    ----------
    parg : Mapping[Any, T] | None
        A mapping of arguments passed as a positional argument.
    kwargs : Mapping[str, T]
        A mapping of arguments passed as keyword arguments.
    func_name : str
        The name of the function for error messages.

    Returns
    -------
    Mapping[Hashable, T]
        A mapping of arguments.
    """
    if parg is None or parg == {}:
        return cast(Mapping[Hashable, T], kwargs)
    if kwargs:
        raise ValueError(
            f'cannot specify both keyword and positional arguments to {func_name}'
        )
    return parg
