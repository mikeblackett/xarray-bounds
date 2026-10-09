"""Global options for xarray bounds.

Shamelessly copied from xarray https://github.com/pydata/xarray/blob/main/xarray/core/options.py
"""

from collections.abc import Hashable
from typing import TypedDict

from xarray.core.utils import FrozenDict

from xarray_bounds.types import ClosedSide, LabelSide

__all__ = ["get_options", "set_options"]


class Options(TypedDict):
    bounds_dim: str
    interval_closed: ClosedSide
    label_side: LabelSide


OPTIONS: Options = {
    "bounds_dim": "bnds",
    "label_side": LabelSide.LEFT,
    "interval_closed": ClosedSide.LEFT,
}

_VALIDATORS = {
    "bounds_dim": lambda value: isinstance(value, Hashable),
    "interval_closed": lambda value: value in ClosedSide,
    "label_side": lambda value: value in LabelSide,
}


class set_options:
    """Set options for :py:mod:`xarray_bounds` in a controlled context."""

    def __init__(self, **kwargs):
        self.old = {}
        for key, value in kwargs.items():
            if key not in OPTIONS:
                raise ValueError(
                    f"argument name {key!r} is not in the set of valid options {set(OPTIONS)!r}"
                )
            if key in _VALIDATORS and not _VALIDATORS[key](value):
                raise ValueError(
                    f"option {key!r} given an invalid value: {value!r}. "
                )
            self.old[key] = OPTIONS[key]
        self._apply_update(kwargs)

    def _apply_update(self, options_dict):
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)


def get_options():
    """
    Get options for xarray_bounds.

    See Also
    --------
    set_options

    """
    return FrozenDict(OPTIONS)
