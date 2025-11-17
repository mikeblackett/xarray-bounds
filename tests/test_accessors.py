import cf_xarray  # noqa: F401
import hypothesis as hp
import hypothesis.strategies as st
import numpy as np
import pytest as pt
import xarray as xr

from xarray_bounds.options import OPTIONS  # noqa: F401

BOUNDS_NAME = OPTIONS['bounds_dim']


class TestDatasetBoundsAccessor:
    """Tests for the DatasetBoundsAccessor class."""

    def test_init(self):
        """Should initialize the bounds accessor."""
        ds = xr.tutorial.open_dataset('air_temperature')
        bounds = ds.bnds
        xr.testing.assert_identical(ds, bounds._obj)

    def test_getitem(self):
        """Should get the bounds for a given dimension/axis."""
        ds = xr.tutorial.open_dataset('air_temperature').cf.add_bounds('time')
        expected = ds['time_bounds']
        actual = ds.bnds['time']
        xr.testing.assert_identical(actual, expected)

    def test_getitem_raises_for_missing_bounds(self):
        """Should raise an error if the bounds are not present for the dimension/axis."""
        ds = xr.tutorial.open_dataset('air_temperature')
        with pt.raises(KeyError):
            ds.bnds['time']

    def test_dims_getter(self):
        """Should return a set of names of dimensions with bounds."""
        dims = ['time', 'lat']
        ds = xr.tutorial.open_dataset('air_temperature').cf.add_bounds(dims)
        expected = set(dims)
        actual = ds.bnds.dims
        assert actual == expected

    def test_iter(self):
        """Should iterate over the bounds."""
        expected = sorted(['time', 'lat'])
        ds = xr.tutorial.open_dataset('air_temperature').cf.add_bounds(
            expected
        )
        actual = sorted(list(iter(ds.bnds)))
        assert actual == expected

    def test_len(self):
        """Should return the number of bounds."""
        dims = sorted(['time', 'lat'])
        ds = xr.tutorial.open_dataset('air_temperature').cf.add_bounds(dims)
        expected = len(dims)
        actual = len(ds.bnds)
        assert actual == expected

    def test_raises_if_add_invalid_key(self):
        """Should raise an error if the key is not resolvable to a known dimension."""
        ds = xr.tutorial.open_dataset('air_temperature')
        with pt.raises(KeyError):
            ds.bnds.infer_bounds('foo')

    def test_raises_for_existing_bounds(self):
        """Should raise an error if the bounds already exist."""
        ds = xr.tutorial.open_dataset('air_temperature').cf.add_bounds('time')
        with pt.raises(ValueError):
            ds.bnds.infer_bounds('time')

    @hp.given(
        key=st.sampled_from(
            [
                # cf axis
                'T',
                # dim name
                'lon',
                # standard_name
                'latitude',
            ]
        )
    )
    def test_infer_bounds_for_key(self, key: str):
        """Should add bounds for the specified dimension.

        The dimension can be specified by any key resolvable by cf-xarray.
        """
        ds = xr.tutorial.open_dataset('air_temperature')
        assert key not in ds.bnds
        actual = ds.bnds.infer_bounds(key)
        assert ds.cf[key].name in actual.bnds

    def test_add_missing_bounds(self):
        """Should add all missing bounds to a DataSet.

        Missing bounds should be added for all dimensions that represent a CF axis.
        """
        ds = xr.tutorial.open_dataset('air_temperature')
        missing = list(ds.dims)
        assert all(d not in ds.bnds for d in missing)
        actual = ds.bnds.infer_bounds()
        assert all(d in actual.bnds for d in missing)

    def test_assign_bounds(self):
        """Should assign new bounds coordinates to this object."""
        ds = xr.tutorial.open_dataset('air_temperature')
        expected = ds.copy().cf.add_bounds('time').time_bounds
        actual = ds.bnds.assign_bounds(time=expected).bnds['time']
        np.testing.assert_array_equal(expected, actual)


class TestDataArrayBoundsAccessor:
    """Tests for the DataArrayBoundsAccessor class."""

    def test_init(self):
        """Should raise a AttributeError."""

        with pt.raises((AttributeError, RuntimeError)):
            xr.DataArray().bounds
