import pytest as pt

import xarray_bounds as xrb

_options = xrb.get_options()


@pt.fixture(autouse=True)
def options():
    with xrb.set_options(**_options):
        yield
