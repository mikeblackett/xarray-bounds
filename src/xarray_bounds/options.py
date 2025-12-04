"""Global options for xarray bounds.

Shamelessly copied from xarray https://github.com/pydata/xarray/blob/main/xarray/core/options.py
"""

from typing import TypedDict

__all__ = ['OPTIONS', 'set_options']


class Options(TypedDict):
    bounds_dim: str


OPTIONS: Options = {
    'bounds_dim': 'bnds',
}


class set_options:
    """Set options for xarray-bounds in a controlled context."""

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(
                    f'argument name {k!r} is not in the set of valid options {set(OPTIONS)!r}'
                )
            self.old[k] = OPTIONS[k]
        self._apply_update(kwargs)

    def _apply_update(self, options_dict):
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)
