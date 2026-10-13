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

from xarray_bounds.options import OPTIONS
from xarray_bounds.types import (
    ClosedSide,
    IntervalClosed,
    IntervalLabel,
    LabelSide,
)


def validate_interval_label(
    label: str | None, default: str | None = None
) -> IntervalLabel:
    if label is not None:
        return LabelSide(label).value
    if label is None and default is None:
        return OPTIONS["label_side"].value
    return LabelSide(default).value


def validate_interval_closed(
    closed: str | None, default: str | None = None
) -> IntervalClosed:
    if closed is not None:
        return ClosedSide(closed).value
    if closed is None and default is None:
        return OPTIONS["interval_closed"].value
    return ClosedSide(default).value


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
        raise KeyError(f"no variable found for key {key!r}.")


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
    alignment: Literal["S", "E"] | None = None
    anchor: str | None = None

    def __str__(self) -> str:
        value = self.base
        if self.n < 0 or self.n > 1:
            value = f"{self.n}{value}"
        if self.alignment:
            value = f"{value}{self.alignment}"
        if self.anchor:
            value = f"{value}-{self.anchor}"
        return value

    @property
    def is_end_aligned(self) -> bool:
        """Whether the offset alias is end-aligned."""
        return self.base == "W" or self.alignment == "E"

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


def _parse_freq(freq: str | pd.DateOffset) -> OffsetAlias:
    """Parse a Pandas frequency string into its components.

    Parameters
    ----------
    freq : str | pd.DateOffset
        A pandas frequency string or DateOffset object.

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
    offset = pd.tseries.frequencies.to_offset(freq)
    alignment = None
    base, *rest = offset.name.split(sep="-")
    if base.endswith(("S", "E")):
        alignment = cast(Literal["S", "E"], base[-1])
        base = base.removesuffix(alignment)
    anchor = rest[0] if rest else None
    return OffsetAlias(
        n=offset.n,
        base=base,
        alignment=alignment,
        anchor=anchor,
    )


T = TypeVar("T")


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
            f"cannot specify both keyword and positional arguments to {func_name}"
        )
    return parg
