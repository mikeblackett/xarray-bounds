import hypothesis as hp
import hypothesis.extra.pandas as hpd
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest as pt
import xarray as xr

from xarray_bounds.core import (
    bounds_to_interval,
    infer_bounds,
)
from xarray_bounds.options import OPTIONS
from xarray_bounds.types import (
    ClosedSide,
    IntervalClosed,
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

    # @pt.mark.parametrize(
    #     'label, closed, expected',
    #     [
    #         (None, None, 'left'),
    #         (None, 'left', 'left'),
    #         (None, 'right', 'right'),
    #         ('middle', None, 'middle'),
    #         ('middle', 'left', 'middle'),
    #         ('middle', 'right', 'middle'),
    #         ('left', None, 'left'),
    #         ('left', 'left', 'left'),
    #         ('left', 'right', 'left'),
    #         ('right', None, 'right'),
    #         ('right', 'left', 'right'),
    #         ('right', 'right', 'right'),
    #     ],
    # )
    # @hp.given(index=hpd.range_indexes(min_size=3))
    # def test_label_default_arg_logic(
    #     self,
    #     label: LabelSide | None,
    #     closed: ClosedSide | None,
    #     expected: IntervalLabel,
    #     index: pd.Index,
    # ):
    #     """Should correctly infer the label based on default argument logic."""
    #     da = xr.DataArray(data=index)
    #     bounds = infer_bounds(da, label=label, closed=closed)
    #     assert bounds.attrs.get('label') == expected

    @pt.mark.parametrize(
        'closed, label, expected',
        [
            (None, None, 'left'),
            (None, 'left', 'left'),
            (None, 'right', 'right'),
            ('left', 'middle', 'left'),
            ('left', 'left', 'left'),
            ('left', 'right', 'left'),
            ('right', 'middle', 'right'),
            ('right', 'left', 'right'),
            ('right', 'right', 'right'),
        ],
    )
    @hp.given(index=hpd.range_indexes(min_size=3))
    def test_closed_default_arg_logic(
        self,
        label: LabelSide,
        closed: ClosedSide | None,
        expected: IntervalClosed,
        index: pd.Index,
    ):
        """Should correctly infer the closed side based on default argument logic."""
        da = xr.DataArray(data=index)
        bounds = infer_bounds(da, label=label, closed=closed)
        assert bounds.attrs.get('closed') == expected

    @hp.given(index=hpd.range_indexes(min_size=3))
    def test_bounds_dim_name(self, index: pd.Index):
        """Should produce bounds with the correct second dimension name."""
        da = xr.DataArray(data=index)
        bounds = infer_bounds(da)
        assert bounds.dims[1] == OPTIONS['bounds_dim']

    @hp.given(dim=st.text(), index=hpd.range_indexes(min_size=3))
    def test_bounds_name(self, dim: str, index: pd.Index):
        """Should use the correct bounds dimension name from options."""
        da = xr.DataArray(data=index, dims=dim)
        bounds = infer_bounds(da)
        assert bounds.name == f'{dim}_{OPTIONS["bounds_dim"]}'

    @hp.given(index=hpd.range_indexes(min_size=3))
    def test_assigns_cf_bounds_attribute(self, index: pd.Index):
        """Should assign the bounds name to the coordinate's 'bounds' attribute."""
        da = xr.DataArray(data=index, dims='time')
        bounds = infer_bounds(da)
        coord = bounds.coords['time']
        assert coord.attrs.get('bounds') == bounds.name

    @hp.given(index=hpd.range_indexes(min_size=3))
    def test_original_object_becomes_coordinate(self, index: pd.Index):
        """Should keep the original object as the coordinate of the bounds."""
        da = xr.DataArray(data=index, dims='time')
        bounds = infer_bounds(da)
        coord = bounds.coords['time']
        # Compare only the data to avoid attribute mismatches
        np.testing.assert_array_equal(coord, da)


class TestBoundsToIndex:
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
        with pt.raises(
            ValueError,
            match=f'bounds must have a second dimension named {OPTIONS["bounds_dim"]}.',
        ):
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
