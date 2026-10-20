from typing import Any

import cf_xarray  # noqa: F401
import hypothesis as hp
import hypothesis.strategies as st
import pytest as pt
import xarray as xr

from xarray_bounds.constants import IntervalLabel
from xarray_bounds.options import OPTIONS

BOUNDS_NAME = OPTIONS["bounds_dim"]

AXES = ["T", "X", "Y"]
NAMES = ["time", "lat", "lon"]
STANDARD_NAMES = ["time", "latitude", "longitude"]
ALIASES = AXES + NAMES + STANDARD_NAMES


class TestClass:
    def test_init(self, ds: xr.Dataset):
        xr.testing.assert_identical(ds, ds.bnds._obj)

    @pt.mark.parametrize("key", ["lat", "latitude", "Y"])
    def test_getitem_with_alias(self, ds: xr.Dataset, key: str):
        expected = ds[f"lat_{BOUNDS_NAME}"]
        actual = ds.bnds[key]
        xr.testing.assert_identical(actual, expected)

    def test_getitem_with_unknown_key_raises_key_error(self, ds: xr.Dataset):
        with pt.raises(KeyError):
            ds.bnds["bogus"]

    @pt.mark.parametrize("key", ["lat", "latitude", "Y"])
    def test_get_with_alias(self, ds: xr.Dataset, key: str):
        expected = ds[f"lat_{BOUNDS_NAME}"]
        actual = ds.bnds.get(key)
        xr.testing.assert_identical(actual, expected)

    @hp.given(expected=st.one_of(st.text(), st.none(), st.integers()))
    def test_get_with_unknown_key_returns_default_arg(self, expected: Any):
        ds = xr.Dataset()
        actual = ds.bnds.get("bogus", default=expected)
        assert actual == expected

    @pt.mark.parametrize("key", ["lat", "latitude", "Y"])
    def test_contains_alias(self, ds: xr.Dataset, key: str):
        assert key in ds.bnds

    def test_contains_unknown_key(self, ds: xr.Dataset):
        assert "bogus" not in ds.bnds

    def test_iter_iterates_canonical_keys(self, ds: xr.Dataset):
        expected = sorted(["time", "lon", "lat"])
        actual = sorted(iter(ds.bnds))
        assert actual == expected

    def test_len_enumerates_canonical_keys(self, ds: xr.Dataset):
        expected = len(ds.cf.axes)
        actual = len(ds.bnds)
        assert actual == expected


class TestBoundsProperties:
    @pt.mark.parametrize("var", ["lat", "lon", "time"])
    def test_variable_names_map_to_bounds_names(
        self, ds: xr.Dataset, var: str
    ):
        expected = f"{var}_{BOUNDS_NAME}"
        assert expected in ds.bnds.variable_names[var]

    @pt.mark.parametrize(
        "var, axis", [("lat", "Y"), ("lon", "X"), ("time", "T")]
    )
    def test_axis_names_map_to_bounds_names(
        self, ds: xr.Dataset, var: str, axis: str
    ):
        expected = f"{var}_{BOUNDS_NAME}"
        assert expected in ds.bnds.axes[axis]

    @pt.mark.parametrize(
        "var, standard_name",
        [("lat", "latitude"), ("lon", "longitude"), ("time", "time")],
    )
    def test_standard_names_map_to_bounds_names(
        self, ds: xr.Dataset, var: str, standard_name: str
    ):
        expected = f"{var}_{BOUNDS_NAME}"
        assert expected in ds.bnds.standard_names[standard_name]

    @pt.mark.parametrize(
        "var, coordinate",
        [("lat", "latitude"), ("lon", "longitude"), ("time", "time")],
    )
    def test_cf_coordinate_names_map_to_bounds_names(
        self, ds: xr.Dataset, var: str, coordinate: str
    ):
        expected = f"{var}_{BOUNDS_NAME}"
        assert expected in ds.bnds.coordinates[coordinate]


class TestAddBounds:
    def test_no_cf_axes_raises_value_error(self, ds_no_bounds: xr.Dataset):
        # should only auto-infer bounds for CF axes
        ds = ds_no_bounds.drop_attrs(deep=True)
        with pt.raises(ValueError, match="guess_coord_axis"):
            ds.bnds.add_bounds()

    def test_invalid_key_raises_key_error(self, ds: xr.Dataset):
        with pt.raises(KeyError):
            ds.bnds.add_bounds("bogus")

    @pt.mark.parametrize("key", ["lat", "longitude", "T"])
    def test_existing_bounds_alias_raises_value_error(
        self, ds: xr.Dataset, key: str
    ):
        with pt.raises(ValueError, match="bounds already exist"):
            ds.bnds.add_bounds(key)

    @pt.mark.parametrize("key", ["lat", "longitude", "T"])
    def test_aliases(self, ds_no_bounds: xr.Dataset, key: str):
        actual = ds_no_bounds.bnds.add_bounds(key)
        assert key in actual.bnds

    def test_no_keys_adds_all_bounds(self, ds_no_bounds: xr.Dataset):
        # Missing bounds should be added for all dimensions that represent a CF axis.
        missing = ["time", "lon", "lat"]
        assert not all(d in ds_no_bounds.bnds for d in missing)
        actual = ds_no_bounds.bnds.add_bounds()
        assert all(d in actual.bnds for d in missing)

    def test_inserts_in_axis_order(self, ds_no_bounds: xr.Dataset):
        expected = list(ds_no_bounds.cf.axes.keys())
        ds_with_bounds = ds_no_bounds.bnds.add_bounds()
        actual = list(ds_with_bounds.bnds.axes.keys())
        assert expected == actual

    @pt.mark.parametrize("label", [None, {}])
    def test_no_label_arg_uses_default(
        self, ds: xr.DataArray, ds_no_bounds: xr.DataArray, label: None | dict
    ):
        actual = ds_no_bounds.bnds.add_bounds(label=label)
        # Should fallback to `datetimeindex_to_interval` default for MS freq (left)
        xr.testing.assert_equal(ds.bnds["time"], actual.bnds["time"])

    def test_no_keys_applies_correct_labels(
        self, ds: xr.DataArray, ds_no_bounds: xr.DataArray
    ):
        label_by_axis = {
            "time": None,
            "lat": IntervalLabel.MIDDLE,
            "lon": IntervalLabel.RIGHT,
        }
        actual = ds_no_bounds.bnds.add_bounds(label=label_by_axis)
        xr.testing.assert_equal(ds.bnds["time"], actual.bnds["time"])
        xr.testing.assert_equal(ds.bnds["lat"], actual.bnds["lat"])
        xr.testing.assert_equal(ds.bnds["lon"], actual.bnds["lon"])

    @pt.mark.parametrize("key", ["Y", "lat", "latitude"])
    def test_label_supports_key_aliases(
        self, ds: xr.DataArray, ds_no_bounds: xr.DataArray, key: str
    ):
        expected = ds.bnds["lat"]
        actual = ds_no_bounds.bnds.add_bounds(
            key, label={key: IntervalLabel.MIDDLE}
        ).bnds["lat"]
        xr.testing.assert_equal(expected, actual)

    def test_unknown_label_key_raises_key_error(
        self, ds_no_bounds: xr.DataArray
    ):
        with pt.raises(KeyError):
            ds_no_bounds.bnds.add_bounds(label={"foo": IntervalLabel.RIGHT})

    def test_invalid_label_value_raises_value_error(
        self, ds_no_bounds: xr.DataArray
    ):
        with pt.raises(ValueError):
            ds_no_bounds.bnds.add_bounds(label={"time": "bogus"})


class TestDropBounds:
    @pt.mark.parametrize("dim", ["time", "T"])
    def test_aliases(self, ds: xr.Dataset, dim: str):
        assert dim in ds.bnds
        actual = ds.bnds.drop_bounds(dim)
        assert dim not in actual.bnds

    def test_no_keys_drops_all_bounds(self, ds: xr.Dataset):
        assert len(ds.bnds) > 0
        actual = ds.bnds.drop_bounds()
        assert len(actual.bnds) == 0

    def test_unknown_alias_raises_key_error(self, ds: xr.Dataset):
        with pt.raises(KeyError, match="no variable"):
            ds.bnds.drop_bounds("bogus")

    def test_unknown_key_raises_key_error(self, ds: xr.Dataset):
        key = "aux"
        assert key in ds
        with pt.raises(KeyError):
            ds.bnds.drop_bounds(key)

    @pt.mark.parametrize("key", ["time", "T", None])
    def test_dangling_attribute_is_removed(
        self, ds: xr.Dataset, key: str | None
    ):
        # resampling drops the bounds variables, but the `bounds` attribute
        # on the new time coordinate survives
        out = ds.resample(time="ME").mean()
        assert f"time_{BOUNDS_NAME}" not in out.variables
        assert out.time.attrs.get("bounds") == f"time_{BOUNDS_NAME}"
        if key is None:
            actual = out.bnds.drop_bounds()
        else:
            actual = out.bnds.drop_bounds(key)
        assert "bounds" not in actual.time.attrs

    @pt.mark.parametrize("key", ["time", "T", None])
    def test_dangling_encoding_is_removed(
        self, ds: xr.Dataset, key: str | None
    ):
        # resampling drops the bounds variables, but the `bounds` attribute
        # on the new time coordinate survives
        out = ds.resample(time="2D").mean()
        # move the dangling reference from attrs to encoding
        del out.time.attrs["bounds"]
        out["time"].encoding["bounds"] = f"time_{BOUNDS_NAME}"
        if key is None:
            actual = out.bnds.drop_bounds()
        else:
            actual = out.bnds.drop_bounds(key)
        assert "bounds" not in actual.time.attrs
        assert "bounds" not in actual["time"].encoding


class TestAssignBounds:
    @pt.mark.parametrize("dim", ["time", "T", "latitude", "Y", "lat"])
    def test_assigns_bounds(
        self, ds: xr.Dataset, ds_no_bounds: xr.Dataset, dim: str
    ):
        ds_new = ds_no_bounds.bnds.assign_bounds(**{dim: ds.bnds[dim]})
        expected = ds.bnds[dim]
        actual = ds_new.bnds[dim]
        xr.testing.assert_equal(expected, actual)

    @pt.mark.parametrize("dim", ["time", "T", "latitude", "Y", "lat"])
    def test_assigns_cf_metadata(
        self, ds: xr.Dataset, ds_no_bounds: xr.Dataset, dim: str
    ):
        ds_new = ds_no_bounds.bnds.assign_bounds(**{dim: ds.bnds[dim]})
        assert ds_new.cf[dim].attrs["bounds"] == ds.bnds[dim].name

    def test_item_assignment_raises_type_error(self, ds: xr.Dataset):
        with pt.raises(TypeError):
            ds.bnds["lat"] = ds[f"lat_{BOUNDS_NAME}"]

    def test_item_deletion_raises_type_error(self, ds: xr.Dataset):
        with pt.raises(TypeError):
            del ds.bnds["lat"]
