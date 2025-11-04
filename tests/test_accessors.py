import cf_xarray  # noqa: F401
import hypothesis as hp
import hypothesis.strategies as st
import pytest as pt
import xarray as xr

import xarray_bounds  # noqa: F401


class TestDatasetBoundsAccessor:
    """Tests for the DatasetBoundsAccessor class."""

    def test_init(self):
        """Should initialize the bounds accessor."""
        ds = xr.tutorial.open_dataset('air_temperature')
        bounds = ds.bounds
        xr.testing.assert_identical(ds, bounds._obj)

    def test_getitem(self):
        """Should get the bounds for a given dimension/axis."""
        ds = xr.tutorial.open_dataset('air_temperature').cf.add_bounds('time')
        expected = ds['time_bounds']
        actual = ds.bounds['time']
        xr.testing.assert_identical(actual, expected)

    def test_getitem_raises_for_missing_bounds(self):
        """Should raise an error if the bounds are not present for the dimension/axis."""
        ds = xr.tutorial.open_dataset('air_temperature')
        with pt.raises(KeyError, match='No bounds found for axis'):
            ds.bounds['time']

    def test_axes_getter(self):
        """Should return a mapping of axes to bounds."""
        ds = xr.tutorial.open_dataset('air_temperature').cf.add_bounds(
            ['time', 'lat', 'lon']
        )
        expected = ds.cf.bounds
        actual = ds.bounds.axes
        for k, v in actual.items():
            assert k in expected
            assert expected[k] == v

    def test_dims_getter(self):
        """Should return a set of names of dimensions with bounds."""
        dims = ['time', 'lat', 'lon']
        ds = xr.tutorial.open_dataset('air_temperature').cf.add_bounds(dims)
        expected = set(dims)
        actual = ds.bounds.dims
        assert actual == expected

    def test_iter(self):
        """Should iterate over the bounds."""
        ds = xr.tutorial.open_dataset('air_temperature').cf.add_bounds(
            ['time', 'lat', 'lon']
        )
        expected = sorted(ds.cf.bounds.keys())
        actual = sorted(list(iter(ds.bounds)))
        assert actual == expected

    def test_len(self):
        """Should return the number of bounds."""
        ds = xr.tutorial.open_dataset('air_temperature').cf.add_bounds(
            ['time', 'lat', 'lon']
        )
        expected = len(ds.cf.bounds)
        actual = len(ds.bounds)
        assert actual == expected

    def test_raises_if_add_invalid_key(self):
        """Should raise an error if the key is not resolvable to a known dimension."""
        ds = xr.tutorial.open_dataset('air_temperature')
        with pt.raises(KeyError):
            ds.bounds.add_bounds('foo')

    def test_raises_for_existing_bounds(self):
        """Should raise an error if the bounds already exist."""
        ds = xr.tutorial.open_dataset('air_temperature').cf.add_bounds('time')
        with pt.raises(ValueError):
            ds.bounds.add_bounds('time')

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
    def test_add_bounds_for_key(self, key: str):
        """Should add bounds for the specified dimension.

        The dimension can be specified by any key resolvable by cf-xarray.
        """
        ds = xr.tutorial.open_dataset('air_temperature')
        assert key not in ds.bounds
        actual = ds.bounds.add_bounds(key)
        assert key in actual.bounds

    def test_add_missing_bounds(self):
        """Should add all missing bounds to a DataSet.

        Missing bounds should be added for all dimensions that represent a CF axis.
        """
        ds = xr.tutorial.open_dataset('air_temperature')
        missing = list(ds.dims)
        assert all(d not in ds.bounds for d in missing)
        actual = ds.bounds.add_bounds()
        assert all(d in actual.bounds for d in missing)

    def test_assign_bounds(self):
        """Should assign new bounds coordinates to this object."""
        ds = xr.tutorial.open_dataset('air_temperature')
        expected = ds.copy().cf.add_bounds('time').time_bounds
        actual = ds.bounds.assign_bounds(time=expected).bounds['time']
        xr.testing.assert_identical(expected, actual)


class TestDataArrayBoundsAccessor:
    """Tests for the DataArrayBoundsAccessor class."""

    def test_init(self):
        """Should raise a AttributeError."""

        with pt.raises((AttributeError, RuntimeError)):
            xr.DataArray().bounds
