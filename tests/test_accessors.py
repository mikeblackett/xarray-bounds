from typing import Any

import cf_xarray  # noqa: F401
import hypothesis as hp
import hypothesis.strategies as st
import numpy as np
import pytest as pt
import xarray as xr

import xarray_bounds as xrb
from xarray_bounds.options import OPTIONS

BOUNDS_NAME = OPTIONS["bounds_dim"]


@pt.fixture(scope="class")
def ds() -> xr.Dataset:
    return xrb.datasets.simple_bounds


@pt.fixture(scope="class")
def ds_no_bounds() -> xr.Dataset:
    return xrb.datasets.simple


class TestDatasetBoundsAccessor:
    def test_init(self, ds: xr.Dataset):
        xr.testing.assert_identical(ds, ds.bnds._obj)

    @pt.mark.parametrize("key", ["lat", "latitude", "Y"])
    def test_getitem(self, ds: xr.Dataset, key: str):
        expected = ds[f"lat_{BOUNDS_NAME}"]
        actual = ds.bnds[key]
        xr.testing.assert_identical(actual, expected)

    def test_getitem_raises_for_invalid_key(self, ds: xr.Dataset):
        with pt.raises(KeyError):
            ds.bnds["foo"]

    @pt.mark.parametrize("key", ["lat", "latitude", "Y"])
    def test_get(self, ds: xr.Dataset, key: str):
        expected = ds[f"lat_{BOUNDS_NAME}"]
        actual = ds.bnds.get(key)
        xr.testing.assert_identical(actual, expected)

    @pt.mark.parametrize("var", ["lat", "lon", "time"])
    def test_variable_names_property(self, ds: xr.Dataset, var: str):
        expected = f"{var}_{BOUNDS_NAME}"
        assert expected in ds.bnds.variable_names[var]

    @pt.mark.parametrize(
        "var, axis", [("lat", "Y"), ("lon", "X"), ("time", "T")]
    )
    def test_axes_property(self, ds: xr.Dataset, var: str, axis: str):
        expected = f"{var}_{BOUNDS_NAME}"
        assert expected in ds.bnds.axes[axis]

    @pt.mark.parametrize(
        "var, standard_name",
        [("lat", "latitude"), ("lon", "longitude"), ("time", "time")],
    )
    def test_standard_names_property(
        self, ds: xr.Dataset, var: str, standard_name: str
    ):
        expected = f"{var}_{BOUNDS_NAME}"
        assert expected in ds.bnds.standard_names[standard_name]

    @pt.mark.parametrize(
        "var, coordinate",
        [("lat", "latitude"), ("lon", "longitude"), ("time", "time")],
    )
    def test_coordinates_property(
        self, ds: xr.Dataset, var: str, coordinate: str
    ):
        expected = f"{var}_{BOUNDS_NAME}"
        assert expected in ds.bnds.coordinates[coordinate]

    @hp.given(value=st.one_of(st.text(), st.none(), st.integers()))
    def test_get_default(self, ds: xr.Dataset, value: Any):
        actual = ds.bnds.get("missing_key", default=value)
        assert actual == value

    @pt.mark.parametrize("key", ["lat", "latitude", "Y"])
    def test_contains(self, ds: xr.Dataset, key: str):
        assert key in ds.bnds

    def test_contains_false(self, ds: xr.Dataset):
        assert "missing_key" not in ds.bnds

    def test_iter(self, ds: xr.Dataset):
        expected = sorted(["time", "lon", "lat"])
        actual = sorted(iter(ds.bnds))
        assert actual == expected

    def test_len(self, ds: xr.Dataset):
        expected = len(ds.cf.axes)
        actual = len(ds.bnds)
        assert actual == expected

    def test_infer_bounds_raises_if_invalid_key(self, ds: xr.Dataset):
        with pt.raises(KeyError):
            ds.bnds.infer_bounds("foo")

    def test_infer_bounds_raises_for_existing_bounds(self, ds: xr.Dataset):
        with pt.raises(ValueError):
            ds.bnds.infer_bounds("time")

    @pt.mark.parametrize(
        "key",
        [
            # cf axis
            "T",
            # dim name
            "lat",
            # standard_name
            "longitude",
        ],
    )
    def test_infer_bounds_for_key(self, ds_no_bounds: xr.Dataset, key: str):
        # Converting the data_var to dataset will drop existing bounds
        actual = ds_no_bounds.bnds.infer_bounds(key)
        assert key in actual.bnds

    def test_infer_all_missing_bounds(self, ds_no_bounds: xr.Dataset):
        # Missing bounds should be added for all dimensions that represent a CF axis.
        missing = ["time", "lon", "lat"]
        assert not all(d in ds_no_bounds.bnds for d in missing)
        actual = ds_no_bounds.bnds.infer_bounds()
        assert all(d in actual.bnds for d in missing)

    @hp.given(dim=st.sampled_from(["time", "T", "latitude", "Y", "lat"]))
    def test_assign_bounds(
        self, ds: xr.Dataset, ds_no_bounds: xr.Dataset, dim: str
    ):
        ds_new = ds_no_bounds.bnds.assign_bounds(**{dim: ds.bnds[dim]})
        expected = ds.bnds[dim]
        actual = ds_new.bnds[dim]
        np.testing.assert_array_equal(expected, actual)

    @hp.given(dim=st.sampled_from(["time", "T", "latitude", "Y", "lat"]))
    def test_assign_bounds_assigns_attribute(
        self, ds: xr.Dataset, ds_no_bounds: xr.Dataset, dim: str
    ):
        ds_new = ds_no_bounds.bnds.assign_bounds(**{dim: ds.bnds[dim]})
        assert ds_new.cf[dim].attrs["bounds"] == ds.bnds[dim].name

    @hp.given(dim=st.sampled_from(["time", "T"]))
    def test_drop_bounds(self, ds: xr.Dataset, dim: str):
        assert dim in ds.bnds
        actual = ds.bnds.drop_bounds(dim)
        assert dim not in actual.bnds
        assert "bounds" not in actual.cf[dim].attrs

    def test_drop_all_bounds(self, ds: xr.Dataset):
        assert len(ds.bnds) > 0
        actual = ds.bnds.drop_bounds()
        assert len(actual.bnds) == 0
        for dim in ["time", "lat", "lon"]:
            assert "bounds" not in actual.cf[dim].attrs

    def test_drop_bounds_raises_if_invalid_key(self, ds: xr.Dataset):
        with pt.raises(KeyError):
            ds.bnds.drop_bounds("foo")

    def test_drop_bounds_removes_cf_bounds_attribute(self, ds: xr.Dataset):
        ds["lat"].encoding["bounds"] = "lat_bounds"  # type: ignore[attr-defined]
        actual = ds.bnds.drop_bounds("lat")
        assert "bounds" not in actual["lat"].attrs
        assert "bounds" not in actual["lat"].encoding

    def test_drop_bounds_removes_cf_bounds_encoding(self, ds: xr.Dataset):
        actual = ds.bnds.drop_bounds("lat")
        assert "bounds" not in actual["lat"].attrs
        assert "bounds" not in actual["lat"].encoding

    def test_raises_for_item_assignment(self, ds: xr.Dataset):
        with pt.raises(TypeError):
            ds.bnds["lat"] = ds[f"lat_{BOUNDS_NAME}"]

    def test_raises_for_item_deletion(self, ds: xr.Dataset):
        with pt.raises(TypeError):
            del ds.bnds["lat"]


class TestDataArrayBoundsAccessor:
    def test_init(self):
        with pt.raises((AttributeError, RuntimeError)):
            xr.DataArray().bounds  # noqa
