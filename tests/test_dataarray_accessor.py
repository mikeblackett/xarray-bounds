import pytest as pt
import xarray as xr


class TestDataArrayBoundsAccessor:
    def test_init(self):
        with pt.raises(NotImplementedError):
            xr.DataArray().bnds  # noqa
