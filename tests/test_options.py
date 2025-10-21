import pytest as pt
import xarray as xr

import xarray_bounds
from xarray_bounds.options import OPTIONS


@pt.mark.parametrize('value', ['bounds', 'bnds'])
def test_set_bounds_dim(value: str):
    """Test the set_options function."""
    xarray_bounds.set_options(bounds_dim=value)
    assert OPTIONS['bounds_dim'] == value


@pt.mark.parametrize('value', ['bounds', 'bnds'])
def test_set_bounds_dim_retention(value: str):
    """Test the set_options function."""
    with xarray_bounds.set_options(bounds_dim=value):
        ds = xr.tutorial.open_dataset('air_temperature').bounds.add_bounds(
            'lat'
        )
        assert value in ds.bounds['lat'].dims
        assert ds.lat.attrs['bounds'] == f'lat_{value}'


def test_invalid_bounds_dim_raises():
    """Setting an invalid bounds_dim should raise a ValueError."""
    with pt.raises(ValueError):
        xarray_bounds.set_options(bounds_dim='foo')


def test_invalid_option_raises():
    """Setting an invalid option should raise a ValueError."""
    with pt.raises(ValueError):
        xarray_bounds.set_options(invalid_option=True)


@pt.mark.parametrize('value', ['bounds', 'bnds'])
def test_get_options(value: str):
    """get_options should return changes made by set_options"""
    with xarray_bounds.set_options(bounds_dim=value):
        options = xarray_bounds.get_options()
        assert options['bounds_dim'] == value
