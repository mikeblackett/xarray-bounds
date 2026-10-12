import hypothesis as hp
import pandas as pd
import pytest as pt
import xarray as xr
from hypothesis import strategies as st

import xarray_bounds as xrb
from tests.constants import FREQ
from xarray_bounds._helpers import (
    OffsetAlias,
    mapping_or_kwargs,
    resolve_variable_name,
    validate_interval_closed,
    validate_interval_label,
)
from xarray_bounds.types import ClosedSide, LabelSide


@pt.fixture(scope="class")
def ds() -> xr.Dataset:
    return xrb.datasets.simple_bounds


class TestValidateIntervalLabel:
    @hp.given(
        label=st.one_of(
            st.none(),
            st.just("invalid_label"),
        ),
        default=st.one_of(
            st.none(),
            st.just("invalid_label"),
        ),
    )
    def test_invalid_argument_raises_value_error(
        self, label: str | None, default: str | None
    ):
        hp.assume(not (label is None and default is None))
        with pt.raises(ValueError):
            validate_interval_label(label=label, default=default)

    @hp.given(
        label=st.one_of(
            st.none(),
            st.sampled_from(LabelSide),
        ),
        default=st.one_of(
            st.none(),
            st.sampled_from(LabelSide),
        ),
    )
    def test_valid_label_is_returned(
        self,
        label: str | None,
        default: str | None,
    ):
        result = validate_interval_label(
            label=label,
            default=default,
        )
        assert result in LabelSide

    @hp.given(
        label=st.one_of(
            st.sampled_from(LabelSide),
        ),
        default=st.one_of(
            st.none(),
            st.sampled_from(LabelSide),
        ),
    )
    def test_valid_label_passes_through(
        self,
        label: str,
        default: str | None,
    ):
        result = validate_interval_label(
            label=label,
            default=default,
        )
        assert result == label

    @hp.given(default=st.sampled_from(LabelSide))
    def test_default_label_is_returned(self, default: LabelSide):
        result = validate_interval_label(label=None, default=default)
        assert result == default.value


class TestValidateIntervalClosed:
    @hp.given(
        closed=st.one_of(
            st.none(),
            st.just("invalid_closed"),
        ),
        default=st.one_of(
            st.none(),
            st.just("invalid_closed"),
        ),
    )
    def test_invalid_argument_raises_value_error(
        self, closed: str | None, default: str | None
    ):
        hp.assume(not (closed is None and default is None))
        with pt.raises(ValueError):
            validate_interval_closed(closed=closed, default=default)

    @hp.given(
        closed=st.one_of(
            st.none(),
            st.sampled_from(ClosedSide),
        ),
        default=st.one_of(
            st.none(),
            st.sampled_from(ClosedSide),
        ),
    )
    def test_valid_interval_closed_is_returned(
        self,
        closed: str | None,
        default: str | None,
    ):
        result = validate_interval_closed(
            closed=closed,
            default=default,
        )
        assert result in ClosedSide

    @hp.given(
        closed=st.one_of(
            st.sampled_from(ClosedSide),
        ),
        default=st.one_of(
            st.none(),
            st.sampled_from(ClosedSide),
        ),
    )
    def test_valid_interval_closed_passes_through(
        self,
        closed: str,
        default: str | None,
    ):
        result = validate_interval_closed(
            closed=closed,
            default=default,
        )
        assert result == closed

    @hp.given(default=st.sampled_from(ClosedSide))
    def test_default_interval_closed_is_returned(self, default: ClosedSide):
        result = validate_interval_closed(closed=None, default=default)
        assert result == default.value


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
        # Normalize `freq` using pandas to_offset to handle equivalent frequency strings
        # i.e. 1MS == MS, YS-JAN == YS, etc.
        date_offset = pd.tseries.frequencies.to_offset(freq)
        offset_alias = OffsetAlias.from_freq(freq)
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
