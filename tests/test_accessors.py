from typing import Literal
import hypothesis as hp
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import xarray as xr
import pytest as pt
import cf_xarray  # noqa: F401
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
        with pt.raises(KeyError):
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
        """Should raise an error if the key is not a known dimension or axis."""
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
        """Should add bounds to a DataSet."""
        ds = xr.tutorial.open_dataset('air_temperature')
        assert key not in ds.bounds
        actual = ds.bounds.add_bounds(key)
        assert key in actual.bounds

    def test_add_missing_bounds(self):
        """Should add all missing bounds to a DataSet."""
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
        """Should raise a NotYetImplementedError."""
        from xarray_bounds.exceptions import NotYetImplementedError

        with pt.raises(NotYetImplementedError):
            xr.DataArray().bounds


class TestDataArrayBndsAccessor:
    """Tests for the DataArrayBndsAccessor class."""

    def test_raises_if_not_2d(self):
        """Should raise an error if the DataArray is not 2D."""
        da = xr.DataArray(range(10))
        with pt.raises(ValueError):
            da.bnds

    def test_raises_if_not_bounds_dim(self):
        """Should raise an error if the DataArray is not 2D."""
        da = xr.DataArray(np.random.rand(10, 10))
        with pt.raises(ValueError):
            da.bnds

    @hp.given(
        start=st.integers(min_value=-10, max_value=10),
        size=st.integers(min_value=3, max_value=10),
        step=st.integers(min_value=1, max_value=3),
        closed=st.sampled_from(['left', 'right']),
    )
    def test_to_interval(
        self,
        start: int,
        size: int,
        step: int,
        closed: Literal['left', 'right'],
    ):
        """Should convert the bounds to a pandas interval index."""
        expected = pd.interval_range(
            start=start,
            end=start + size * step,
            freq=step,
            closed=closed,
        )
        index = expected.left
        ds = xr.Dataset(
            data_vars={'foo': ('lat', range(len(index)))},
            coords={'lat': index},
        ).bounds.add_bounds('lat', closed=closed)
        actual = ds.bounds['lat'].bnds.to_interval()
        np.testing.assert_array_equal(actual=actual, desired=expected)

    @hp.given(
        start=st.integers(min_value=-10, max_value=10),
        size=st.integers(min_value=3, max_value=10),
        step=st.integers(min_value=1, max_value=3),
        closed=st.sampled_from(['left', 'right']),
    )
    def test_midpoint(
        self,
        start: int,
        size: int,
        step: int,
        closed: Literal['left', 'right'],
    ):
        """Should return the midpoint of the bounds."""
        end = start + size * step
        interval = pd.interval_range(
            start=start, end=end, freq=step, closed=closed
        )
        expected = interval.mid
        ds = xr.Dataset(
            data_vars={'foo': ('lat', range(len(interval)))},
            coords={'lat': interval.left},
        ).bounds.add_bounds('lat', closed=closed)
        actual = ds.bounds['lat'].bnds.midpoint
        np.testing.assert_array_equal(actual=actual, desired=expected)

    @hp.given(
        start=st.integers(min_value=-10, max_value=10),
        size=st.integers(min_value=3, max_value=10),
        step=st.integers(min_value=1, max_value=3),
        closed=st.sampled_from(['left', 'right']),
    )
    def test_length(
        self,
        start: int,
        size: int,
        step: int,
        closed: Literal['left', 'right'],
    ):
        """Should return the midpoint of the bounds."""
        end = start + size * step
        interval = pd.interval_range(
            start=start, end=end, freq=step, closed=closed
        )
        expected = interval.length
        ds = xr.Dataset(
            data_vars={'foo': ('lat', range(len(interval)))},
            coords={'lat': interval.left},
        ).bounds.add_bounds('lat', closed=closed)
        actual = ds.bounds['lat'].bnds.length
        np.testing.assert_array_equal(actual=actual, desired=expected)
