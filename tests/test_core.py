import hypothesis as hp
import hypothesis.extra.pandas as hpd
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest as pt
import xarray as xr
import xarray.testing.strategies as xrst

from xarray_bounds.core import (
    bounds_to_interval,
    infer_bounds,
    interval_to_bounds,
)
from xarray_bounds.options import OPTIONS, set_options
from xarray_bounds.types import (
    ClosedSide,
    LabelSide,
)


class TestInferBounds:
    def test_raises_if_object_not_1d(self):
        """Should raise an error if the object is not 1D."""

        da = xr.tutorial.open_dataset('air_temperature').air
        with pt.raises(
            ValueError,
            match='1-dimensional',
        ):
            infer_bounds(da)

    @hp.given(index=hpd.range_indexes(min_size=3))
    def test_returns_array_with_shape_n_2(self, index: pd.Index):
        """Should return an array with shape (n, 2).

        CF Conventions 7.1.1:
        "For a one-dimensional coordinate variable of size N, the boundary
        variable is an array of shape (N,2)."
        """
        da = xr.DataArray(data=index, dims='dim')
        bounds = infer_bounds(da)
        n = len(da)
        assert bounds.shape == (n, 2)

    @hp.given(index=hpd.range_indexes(min_size=3))
    def test_returns_cf_compliant_bounds(self, index: pd.Index):
        """Should return bounds that comply with CF conventions.

        CF Conventions 7.1.1:
        "The bounds for cell i are the elements B(i,0) and B(i,1) of the boundary variable B.
        Element C(i) of the coordinate variable C should lie between the boundaries of the cell, or upon one of them
        i.e. B(i,0) - C(i) and B(i,1) - C(i) should not have the same sign,"
        """
        da = xr.DataArray(data=index, dims='dim')
        bounds = infer_bounds(da)
        for i in range(len(da)):
            lower_diff = bounds[i, 0] - da[i]
            upper_diff = bounds[i, 1] - da[i]
            assert not (np.sign(lower_diff) == np.sign(upper_diff))

    @hp.given(index=hpd.range_indexes(min_size=3))
    def test_returns_array_with_attrs(self, index: pd.Index):
        """Should return an array with 'closed' and 'label' attributes."""
        da = xr.DataArray(data=index, dims='dim')
        bounds = infer_bounds(da)
        assert 'closed' in bounds.attrs

    @hp.given(index=hpd.range_indexes(min_size=3))
    def test_bounds_dim(self, index: pd.Index):
        """Should produce bounds with the correct second dimension."""
        da = xr.DataArray(data=index)
        bounds = infer_bounds(da)
        assert bounds.dims[1] == OPTIONS['bounds_dim']

    @hp.given(dim=xrst.names(), index=hpd.range_indexes(min_size=3))
    def test_bounds_name(self, dim: str, index: pd.Index):
        """Should produce bounds with the correct variable name."""
        da = xr.DataArray(data=index, dims=dim)
        bounds = infer_bounds(da)
        assert bounds.name == f'{dim}_{OPTIONS["bounds_dim"]}'

    @hp.given(index=hpd.range_indexes(min_size=3))
    def test_original_object_is_assigned_as_coordinate(self, index: pd.Index):
        """Should keep the original object as the coordinate of the bounds."""
        da = xr.DataArray(data=index, dims='time')
        bounds = infer_bounds(da)
        coord = bounds.coords['time']
        # Compare only the data to avoid attribute mismatches
        np.testing.assert_array_equal(coord, da)

    @hp.given(
        index=hpd.range_indexes(min_size=3, name=xrst.names()),
        name=st.one_of(xrst.names(), st.none()),
    )
    def test_coordinate_name(self, index: pd.Index, name: str | None):
        """Should produce bounds with the correct coordinate name.

        The coordinate name should match the original object's name,
        or the dimension name if the original object has no name.
        """
        da = xr.DataArray(data=index, name=name)
        bounds = infer_bounds(da)
        if name is None:
            assert bounds.coords[index.name].name == index.name
        else:
            assert bounds.coords[name].name == name

    @hp.given(index=hpd.range_indexes(min_size=3))
    def test_assigns_cf_bounds_attribute(self, index: pd.Index):
        """Should assign the bounds name to the coordinate's attributes."""
        da = xr.DataArray(data=index, dims='time')
        bounds = infer_bounds(da)
        coord = bounds.coords['time']
        assert coord.attrs.get('bounds') == bounds.name


class TestIntervalToBounds:
    @hp.given(closed=st.sampled_from(['both', 'neither']))
    def test_raises_if_invalid_closed_attribute(self, closed: str):
        """Should raise an error if the object is not 2D."""
        interval = pd.interval_range(0, 10, periods=5, closed=closed)
        with pt.raises(ValueError):
            interval_to_bounds(interval)

    def test_raises_if_no_dim_and_no_name(self):
        """Should raise an error if no dim is provided and interval has no name."""
        interval = pd.interval_range(0, 10, periods=5)
        with pt.raises(ValueError):
            interval_to_bounds(interval)

    def test_returns_data_array_with_correct_shape(self):
        """Should return an xarray DataArray."""
        interval = pd.interval_range(
            0, 10, periods=5, closed='left', name='interval'
        )
        bounds = interval_to_bounds(interval)
        assert bounds.shape == (5, 2)

    def test_returns_data_array_with_correct_dims(self):
        """Should return an xarray DataArray with correct dimensions."""
        interval = pd.interval_range(
            0, 10, periods=5, closed='right', name='interval'
        )
        bounds = interval_to_bounds(interval, dim='my_dim')
        assert bounds.dims == ('my_dim', OPTIONS['bounds_dim'])

    @pt.mark.parametrize(
        'index_name, dim, name, expected',
        [
            ('interval', None, None, 'interval'),
            ('interval', 'dim', None, 'dim'),
            ('interval', 'dim', 'name', 'name'),
        ],
    )
    def test_returns_data_array_with_correct_coordinate_name(
        self, index_name, dim, name, expected
    ):
        """Should handle naming of axis coordinate correctly."""
        interval = pd.interval_range(
            0, 10, periods=5, closed='right', name=index_name
        )
        bounds = interval_to_bounds(interval, dim=dim, name=name)

        assert expected in bounds.coords

    @pt.mark.parametrize(
        'index_name, dim, name, expected',
        [
            ('interval', None, None, 'interval'),
            ('interval', 'dim', None, 'dim'),
            ('interval', 'dim', 'name', 'dim'),
        ],
    )
    def test_returns_data_array_with_correct_dimension_name(
        self, index_name, dim, name, expected
    ):
        """Should handle naming of axis coordinate correctly."""
        interval = pd.interval_range(
            0, 10, periods=5, closed='right', name=index_name
        )
        bounds = interval_to_bounds(interval, dim=dim, name=name)

        assert expected in bounds.dims

    @pt.mark.parametrize(
        'index_name, dim, name, expected',
        [
            ('interval', None, None, 'interval_bnds'),
            ('interval', 'dim', None, 'dim_bnds'),
            ('interval', 'dim', 'name', 'name_bnds'),
        ],
    )
    def test_returns_data_array_with_correct_bounds_name(
        self, index_name, dim, name, expected
    ):
        """Should handle naming of axis coordinate correctly."""
        with set_options(bounds_dim='bnds'):
            interval = pd.interval_range(
                0, 10, periods=5, closed='right', name=index_name
            )
            bounds = interval_to_bounds(interval, dim=dim, name=name)

            assert bounds.name == expected

    @hp.given(closed=st.sampled_from(ClosedSide))
    def test_adds_closed_attribute(self, closed: str):
        """Should add the closed side to the coordinate attributes."""
        interval = pd.interval_range(
            0, 10, periods=5, closed=closed, name='test'
        )

        bounds = interval_to_bounds(interval)
        assert (
            bounds.coords['test'].attrs['bounds']
            == f'test_{OPTIONS["bounds_dim"]}'
        )

    @hp.given(name=xrst.names())
    def test_adds_bounds_attribute(self, name: str):
        """Should add the bounds attribute to the coordinate."""
        interval = pd.interval_range(
            0, 10, periods=5, closed='right', name=name
        )
        bounds = interval_to_bounds(interval)
        coord = bounds.coords[name]
        assert coord.attrs['bounds'] == bounds.name

    @pt.mark.parametrize(
        'data',
        [
            [(0, 1), (1, 2), (2, 3)],
        ],
    )
    def test_bounds_correctness(self, data: list):
        """Should produce correct bounds data."""
        interval = pd.IntervalIndex.from_tuples(data, name='test')
        bounds = interval_to_bounds(interval)
        expected = np.array(data)
        np.testing.assert_array_equal(bounds.data, expected)

    @pt.mark.parametrize(
        'data, label, expected',
        [
            ([(0, 1), (1, 2), (2, 3)], 'left', [0, 1, 2]),
            ([(0, 1), (1, 2), (2, 3)], 'right', [1, 2, 3]),
            ([(0, 1), (1, 2), (2, 3)], 'middle', [0.5, 1.5, 2.5]),
        ],
    )
    def test_coord_correctness(self, data: list, label: str, expected: list):
        """Should keep the original intervals as the coordinate."""
        interval = pd.IntervalIndex.from_tuples(data, name='test')
        bounds = interval_to_bounds(interval, label=label)
        coord = bounds.coords['test']
        np.testing.assert_array_equal(coord.data, expected)


class TestBoundsToInterval:
    def test_raises_if_not_2d(self):
        """Should raise an error if the dimension is not 2D."""
        da = xr.tutorial.open_dataset('air_temperature').time
        with pt.raises(ValueError, match='bounds must be a 2D DataArray'):
            bounds_to_interval(da)

    def test_raises_if_wrong_bounds_dim(self):
        """Should raise an error if the second dimension is not the bounds dim."""
        da = xr.tutorial.open_dataset('air_temperature').time
        bounds = infer_bounds(da)
        bounds = bounds.rename({OPTIONS['bounds_dim']: 'wrong_dim'})
        with pt.raises(ValueError):
            bounds_to_interval(bounds)

    @hp.given(index=hpd.range_indexes(min_size=3))
    def test_returns_interval_index(self, index: pd.Index):
        """Should return a pandas IntervalIndex."""
        da = xr.DataArray(data=index)
        bounds = infer_bounds(da)
        interval_index = bounds_to_interval(bounds)
        assert isinstance(interval_index, pd.IntervalIndex)

    @hp.given(
        index=hpd.range_indexes(min_size=3),
        closed=st.sampled_from(ClosedSide),
        label=st.sampled_from(LabelSide),
    )
    def test_interval_index_correctness(
        self, index: pd.Index, closed: ClosedSide, label: LabelSide
    ):
        """Should return an IntervalIndex that matches the inferred bounds."""
        da = xr.DataArray(data=index)
        bounds = infer_bounds(da, closed=closed, label=label)
        interval = bounds_to_interval(bounds)
        for i in range(len(da)):
            assert interval[i].left == bounds[i, 0]
            assert interval[i].right == bounds[i, 1]
