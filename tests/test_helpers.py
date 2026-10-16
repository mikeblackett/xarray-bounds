import hypothesis as hp
import pandas as pd
import pytest as pt
import xarray as xr
from hypothesis import strategies as st

from tests.constants import FREQ
from xarray_bounds._helpers import (
    OffsetAlias,
    mapping_or_kwargs,
    resolve_axis_name,
    resolve_variable_name,
)


class TestResolveAxisName:
    @pt.mark.parametrize(
        "key",
        [
            # Coordinate that exists but is not an axis
            "aux",
            # Non-existent coordinate
            "foo",
        ],
    )
    def test_invalid_axis_name_raises_key_error(
        self, ds: xr.Dataset, key: str
    ):
        with pt.raises(KeyError):
            resolve_axis_name(ds, key)

    @pt.mark.parametrize(
        "key",
        [
            # CF standard_name
            "longitude",
            # CF axis name
            "X",
            # variable name
            "lon",
        ],
    )
    def test_axis_name_is_resolved(self, ds: xr.Dataset, key: str):
        assert resolve_axis_name(ds, key) == "X"


class TestResolveVariableName:
    def test_invalid_variable_name_raises_key_error(self, ds: xr.Dataset):
        with pt.raises(KeyError):
            resolve_variable_name(ds, "foo")

    @pt.mark.parametrize(
        "key",
        [
            # CF standard_name
            "latitude",
            # CF axis name
            "Y",
            # variable name
            "lat",
        ],
    )
    def test_variable_name_is_resolved(self, ds: xr.Dataset, key: str):
        assert resolve_variable_name(ds, key) == "lat"


class TestOffsetAlias:
    @hp.given(freq=st.sampled_from(FREQ))
    def test_constructs_from_freq_string(self, freq: str):
        offset_alias = OffsetAlias.from_freq(freq)
        # For comparisons: normalize `freq` using pandas to_offset to handle equivalent frequency strings
        # i.e. 1MS == MS, YS-JAN == YS, etc.
        date_offset = pd.tseries.frequencies.to_offset(freq)
        assert isinstance(offset_alias, OffsetAlias)
        assert offset_alias.freqstr == date_offset.freqstr

    @hp.given(freq=st.sampled_from(FREQ))
    def test_constructs_from_date_offset(self, freq: str):
        date_offset = pd.tseries.frequencies.to_offset(freq)
        offset_alias = OffsetAlias.from_freq(date_offset)  # type: ignore
        assert isinstance(offset_alias, OffsetAlias)
        assert offset_alias.freqstr == date_offset.freqstr

    @hp.given(freq=st.sampled_from(FREQ))
    def test_freq_str_roundtrips(self, freq: str):
        offset_alias = OffsetAlias.from_freq(freq)
        # convert to offset to normalize freq string with default anchors:
        # i.e. YS == YS-JAN
        expected = pd.tseries.frequencies.to_offset(freq)
        observed = pd.tseries.frequencies.to_offset(offset_alias.freqstr)
        assert expected == observed

    @hp.given(freq=st.sampled_from(FREQ))
    def test_offset_roundtrips(self, freq: str):
        offset = pd.tseries.frequencies.to_offset(freq)
        offset_alias = OffsetAlias.from_freq(offset)  # type: ignore
        assert offset_alias.offset == offset

    @pt.mark.parametrize(
        "freq, expected",
        [
            ("MS", False),
            ("QE", True),
            ("YS-FEB", False),
            ("QE-DEC", True),
            ("W", True),
            ("W-TUE", True),
        ],
    )
    def test_end_aligned_frequencies(self, freq: str, expected: bool):
        offset = OffsetAlias.from_freq(freq)

        assert offset.is_end_aligned is expected


class TestMappingOrKwargs:
    @pt.mark.parametrize(
        "pargs, kwargs, func_name, expected",
        [
            ({"a": 1, "b": 2}, {}, "test_func", {"a": 1, "b": 2}),
            (None, {"a": 1, "b": 2}, "test_func", {"a": 1, "b": 2}),
            ({}, {"a": 1, "b": 2}, "test_func", {"a": 1, "b": 2}),
        ],
    )
    def test_mapping_or_kwargs(
        self,
        pargs,
        kwargs,
        func_name,
        expected,
    ):
        result = mapping_or_kwargs(pargs, kwargs, func_name)
        assert result == expected

    @pt.mark.parametrize(
        "pargs, kwargs, func_name",
        [
            ({"a": 1}, {"b": 2}, "test_func"),
        ],
    )
    def test_passing_parg_and_kwargs_raises_value_error(
        self, pargs, kwargs, func_name
    ):
        with pt.raises(ValueError):
            mapping_or_kwargs(pargs, kwargs, func_name)
