import pytest as pt
import xarray as xr

import xarray_bounds as xrb

_options = xrb.get_options()


@pt.fixture(autouse=True)
def options():
    with xrb.set_options(**_options):
        yield


@pt.fixture
def ds(options) -> xr.Dataset:
    return xrb.datasets.make_tiny_dataset()


@pt.fixture
def ds_no_bounds(options) -> xr.Dataset:
    ds = xrb.datasets.make_tiny_dataset().tas.to_dataset()
    for da in ds.coords.values():
        if "bounds" in da.attrs:
            del da.attrs["bounds"]
    return ds
