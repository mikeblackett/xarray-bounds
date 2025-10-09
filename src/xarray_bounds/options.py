"""Global options for xarray bounds.

Shamelessly stolen from xarray https://github.com/pydata/xarray/blob/main/xarray/core/options.py
"""

from typing import TypedDict

from xarray.core.utils import FrozenDict


class Options(TypedDict):
    bounds_name: str


OPTIONS: Options = {
    'bounds_name': 'bounds',
}

_BOUNDS_NAME_OPTIONS = ['bounds', 'bnds']

_VALIDATORS = {
    'bounds_name': lambda v: v in _BOUNDS_NAME_OPTIONS,
}

_SETTERS = {}


class set_options:
    """Set options for xarray in a controlled context."""

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(
                    f'argument name {k!r} is not in the set of valid options {set(OPTIONS)!r}'
                )
            if k in _VALIDATORS and not _VALIDATORS[k](v):
                expected = ''
                if k == 'bounds_name':
                    expected = f'Expected one of {_BOUNDS_NAME_OPTIONS!r}'
                raise ValueError(
                    f'option {k!r} given an invalid value: {v!r}. ' + expected
                )
            self.old[k] = OPTIONS[k]
        self._apply_update(kwargs)

    def _apply_update(self, options_dict):
        for k, v in options_dict.items():
            if k in _SETTERS:
                _SETTERS[k](v)
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)


def get_options():
    """
    Get options for xarray bounds.

    See Also
    ----------
    set_options

    """
    return FrozenDict(OPTIONS)
