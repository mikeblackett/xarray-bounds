from typing import Literal

import hypothesis as hp
import hypothesis.extra.pandas as hpd
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest as pt
import xarray as xr
import xarray.testing.strategies as xrst

from xarray_bounds.constants import (
    IntervalClosed,
    IntervalLabel,
)
from xarray_bounds.core import (
    bounds_to_interval,
    infer_bounds,
    interval_to_bounds,
)
from xarray_bounds.options import OPTIONS, set_options
from xarray_bounds.utilities import datetime_to_interval

rng = np.random.default_rng(12345)


class TestInferBounds:
    def test_mutli_dimensional_index_raises_value_error(self):
        da = xr.DataArray.from_dict(
            {"dims": ("time", "vertical"), "data": rng.lognormal(size=(2, 2))}
        )
        with pt.raises(ValueError, match="1-dimensional"):
            infer_bounds(da)

    @hp.given(index=hpd.range_indexes(min_size=3))
    def test_bounds_shape(self, index: pd.Index):
        # CF Conventions 7.1.1:
        # "For a one-dimensional coordinate variable of size N, the boundary
        # variable is an array of shape (N,2)."
        da = xr.DataArray(data=index, dims="dim")
        expected = (len(da), 2)
        actual = infer_bounds(da).shape
        assert actual == expected

    @hp.given(index=hpd.range_indexes(min_size=3))
    def test_bounds_are_cf_compliant(self, index: pd.Index):
        # CF Conventions 7.1.1:
        # "The bounds for cell i are the elements B(i,0) and B(i,1) of the boundary variable B.
        # Element C(i) of the coordinate variable C should lie between the boundaries of the cell, or upon one of them
        # i.e. B(i,0) - C(i) and B(i,1) - C(i) should not have the same sign,"
        da = xr.DataArray(data=index, dims="dim")
        bounds = infer_bounds(da)
        for i in range(len(da)):
            lower_diff = bounds[i, 0] - da[i]
            upper_diff = bounds[i, 1] - da[i]
            assert np.sign(lower_diff) != np.sign(upper_diff)

    @hp.given(index=hpd.range_indexes(min_size=3))
    def test_closed_attr(self, index: pd.Index):
        da = xr.DataArray(data=index, dims="dim")
        bounds = infer_bounds(da)
        assert "closed" in bounds.attrs

    @hp.given(index=hpd.range_indexes(min_size=3))
    def test_cf_bounds_attr(self, index: pd.Index):
        da = xr.DataArray(data=index, dims="time")
        bounds = infer_bounds(da)
        coord = bounds.coords["time"]
        assert coord.attrs.get("bounds") == bounds.name

    @hp.given(index=hpd.range_indexes(min_size=3))
    def test_variable_name(self, index: pd.Index):
        da = xr.DataArray(data=index)
        bounds = infer_bounds(da)
        assert bounds.dims[1] == OPTIONS["bounds_dim"]

    @hp.given(dim=xrst.names(), index=hpd.range_indexes(min_size=4))
    def test_dim_name(self, dim: str, index: pd.Index):
        hp.assume(dim != OPTIONS["bounds_dim"])
        da = xr.DataArray(data=index, dims=dim)
        bounds = infer_bounds(da)
        assert bounds.name == f"{dim}_{OPTIONS['bounds_dim']}"

    @hp.given(index=hpd.range_indexes(min_size=3))
    def test_coordinate_equality(self, index: pd.Index):
        da = xr.DataArray(data=index, dims="time")
        bounds = infer_bounds(da)
        coord = bounds.coords["time"]
        # Compare only the data to avoid attribute mismatches
        np.testing.assert_array_equal(coord, da)

    @hp.given(
        index=hpd.range_indexes(min_size=3, name=xrst.names()),
        name=st.one_of(xrst.names(), st.none()),
    )
    def test_coordinate_name(self, index: pd.Index, name: str | None):
        # The coordinate name should match the original object's name,
        # or the dimension name if the original object has no name.
        hp.assume(name != OPTIONS["bounds_dim"])
        da = xr.DataArray(data=index, name=name)
        bounds = infer_bounds(da)
        if name is None:
            assert bounds.coords[index.name].name == index.name
        else:
            assert bounds.coords[name].name == name

    @hp.given(
        index=hpd.range_indexes(
            min_size=3, name=st.just(OPTIONS["bounds_dim"])
        ),
    )
    def test_name_conflict_raises_value_error(self, index: pd.Index):
        # the variable name can't be the same as the bounds dimension name
        da = xr.DataArray(data=index)
        with pt.raises(ValueError):
            infer_bounds(da)

    def test_datetime_end_aligned_defaults_to_right_interval(self):
        # Regression test: end-aligned frequencies (e.g. ME) must default
        # to right labels, so the bounds end on the index values.
        index = pd.date_range("2000-01-31", periods=3, freq="ME")
        da = xr.DataArray(data=index, dims="time")
        bounds = infer_bounds(da)
        expected = datetime_to_interval(index)
        actual = bounds_to_interval(bounds)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual.right, index)
        assert bounds.attrs["closed"] == "right"

    def test_datetime_start_aligned_defaults_to_left_interval(self):
        index = pd.date_range("2000-01-01", periods=3, freq="MS")
        da = xr.DataArray(data=index, dims="time")
        bounds = infer_bounds(da)
        expected = datetime_to_interval(index)
        actual = bounds_to_interval(bounds)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual.left, index)
        assert bounds.attrs["closed"] == "left"

    @pt.mark.parametrize("label", IntervalLabel)
    def test_label_without_closed_does_not_raise(self, label: IntervalLabel):
        # `closed` is independent of `label`; omitting it must not raise,
        # even when `label='middle'` is not a valid closed side.
        index = pd.date_range("2000-01-01", periods=4, freq="D")
        da = xr.DataArray(data=index, dims="time")
        bounds = infer_bounds(da, label=label)
        assert bounds.attrs["closed"] == "left"

    def test_invalid_label_raises_value_error(self):
        da = xr.DataArray(data=range(3), dims="x")
        with pt.raises(ValueError):
            infer_bounds(da, label="bogus")  # type: ignore


class TestIntervalToBounds:
    @hp.given(closed=st.sampled_from(["both", "neither"]))
    def test_invalid_closed_raises_value_error(
        self, closed: Literal["both", "neither"]
    ):
        interval = pd.interval_range(0, 10, periods=5, closed=closed)
        with pt.raises(ValueError):
            interval_to_bounds(interval)

    def test_interval_with_no_name_or_dim_raises_value_error(self):
        interval = pd.interval_range(0, 10, periods=5)
        with pt.raises(ValueError):
            interval_to_bounds(interval)

    def test_dimension_name_conflict_raises_value_error(self):
        bounds_dim_name = OPTIONS["bounds_dim"]
        interval = pd.interval_range(
            0, 10, periods=5, closed="right", name=bounds_dim_name
        )
        with pt.raises(ValueError):
            interval_to_bounds(interval)

    def test_bounds_shape(self):
        interval = pd.interval_range(
            0, 10, periods=5, closed="left", name="interval"
        )
        bounds = interval_to_bounds(interval)
        assert bounds.shape == (5, 2)

    def test_bounds_dims(self):
        interval = pd.interval_range(
            0, 10, periods=5, closed="right", name="interval"
        )
        bounds = interval_to_bounds(interval, dim="my_dim")
        assert bounds.dims == ("my_dim", OPTIONS["bounds_dim"])

    @pt.mark.parametrize(
        "index_name, dim, name, expected",
        [
            ("interval", None, None, "interval"),
            ("interval", "dim", None, "dim"),
            ("interval", "dim", "name", "name"),
        ],
    )
    def test_coordinate_assigned(self, index_name, dim, name, expected):
        interval = pd.interval_range(
            0, 10, periods=5, closed="right", name=index_name
        )
        bounds = interval_to_bounds(interval, dim=dim, name=name)
        assert expected in bounds.coords

    @pt.mark.parametrize(
        "index_name, dim, name, expected",
        [
            ("interval", None, None, "interval"),
            ("interval", "dim", None, "dim"),
            ("interval", "dim", "name", "dim"),
        ],
    )
    def test_dim_name(self, index_name, dim, name, expected):
        interval = pd.interval_range(
            0, 10, periods=5, closed="right", name=index_name
        )
        bounds = interval_to_bounds(interval, dim=dim, name=name)

        assert expected in bounds.dims

    @pt.mark.parametrize(
        "index_name, dim, name, expected",
        [
            ("interval", None, None, "interval_bnds"),
            ("interval", "dim", None, "dim_bnds"),
            ("interval", "dim", "name", "name_bnds"),
        ],
    )
    def test_variable_name(self, index_name, dim, name, expected):
        with set_options(bounds_dim="bnds"):
            interval = pd.interval_range(
                0, 10, periods=5, closed="right", name=index_name
            )
            bounds = interval_to_bounds(interval, dim=dim, name=name)

            assert bounds.name == expected

    @hp.given(closed=st.sampled_from(IntervalClosed))
    def test_closed_attribute(self, closed: IntervalClosed):
        interval = pd.interval_range(
            0, 10, periods=5, closed=closed.value, name="test"
        )

        bounds = interval_to_bounds(interval)
        assert (
            bounds.coords["test"].attrs["bounds"]
            == f"test_{OPTIONS['bounds_dim']}"
        )

    @hp.given(name=xrst.names())
    def test_bounds_attribute(self, name: str):
        hp.assume(name != OPTIONS["bounds_dim"])
        interval = pd.interval_range(
            0, 10, periods=5, closed="right", name=name
        )
        bounds = interval_to_bounds(interval)
        coord = bounds.coords[name]
        assert coord.attrs["bounds"] == bounds.name

    @pt.mark.parametrize(
        "data",
        [
            [(0, 1), (1, 2), (2, 3)],
        ],
    )
    def test_computes_bounds(self, data: list):
        interval = pd.IntervalIndex.from_tuples(data, name="test")
        bounds = interval_to_bounds(interval)
        expected = np.array(data)
        np.testing.assert_array_equal(bounds.data, expected)

    @pt.mark.parametrize(
        "data, label, expected",
        [
            ([(0, 1), (1, 2), (2, 3)], "left", [0, 1, 2]),
            ([(0, 1), (1, 2), (2, 3)], "right", [1, 2, 3]),
            ([(0, 1), (1, 2), (2, 3)], "middle", [0.5, 1.5, 2.5]),
        ],
    )
    def test_coordinate_equality(
        self, data: list, label: IntervalLabel, expected: list
    ):
        interval = pd.IntervalIndex.from_tuples(data, name="test")
        bounds = interval_to_bounds(interval, label=label)
        coord = bounds.coords["test"]
        np.testing.assert_array_equal(coord.data, expected)


class TestBoundsToInterval:
    def test_missing_closed_attribute_defaults_to_left(self):
        da = xr.DataArray(data=range(3), dims="x")
        bounds = infer_bounds(da)
        del bounds.attrs["closed"]
        interval = bounds_to_interval(bounds)
        assert interval.closed == "left"

    def test_2_dimensional_array_raises_value_error(self):
        da = xr.DataArray.from_dict({"dims": "time", "data": range(10)})
        with pt.raises(ValueError, match="2-dimensional"):
            bounds_to_interval(da)

    @hp.given(index=hpd.range_indexes(min_size=3))
    def test_return_type_is_interval_index(self, index: pd.Index):
        da = xr.DataArray(data=index)
        bounds = infer_bounds(da)
        interval_index = bounds_to_interval(bounds)
        assert isinstance(interval_index, pd.IntervalIndex)

    @hp.given(
        index=hpd.range_indexes(min_size=3),
        closed=st.sampled_from(IntervalClosed),
        label=st.sampled_from(IntervalLabel),
    )
    def test_computes_intervals(
        self, index: pd.Index, closed: IntervalClosed, label: IntervalLabel
    ):
        da = xr.DataArray(data=index)
        bounds = infer_bounds(da, closed=closed, label=label)
        interval = bounds_to_interval(bounds)
        for i in range(len(da)):
            assert interval[i].left == bounds[i, 0]
            assert interval[i].right == bounds[i, 1]
