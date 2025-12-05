import hypothesis as hp
import pandas as pd
import pytest as pt
import xarray as xr
import xarray_strategies as xrst
from hypothesis import strategies as st

import xarray_bounds as xrb
from xarray_bounds.helpers import (
    OffsetAlias,
    mapping_or_kwargs,
    resolve_axis_name,
    resolve_standard_name,
    resolve_variable_name,
    validate_interval_closed,
    validate_interval_label,
)
from xarray_bounds.types import ClosedSide, LabelSide


@pt.fixture(scope='class')
def ds() -> xr.Dataset:
    return xrb.datasets.simple_bounds


class TestValidateIntervalLabel:
    @hp.given(
        label=st.one_of(
            st.none(),
            st.just('invalid_label'),
        ),
        default=st.one_of(
            st.none(),
            st.just('invalid_label'),
        ),
    )
    def test_should_raise_value_error_for_invalid_arguments(
        self, label: str | None, default: str | None
    ):
        """Should raise a ValueError for invalid IntervalLabel."""
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
    def test_returns_valid_interval_label(
        self,
        label: str | None,
        default: str | None,
    ):
        """Should return a valid IntervalLabel."""
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
    def test_returns_label_if_label_is_not_none(
        self,
        label: str,
        default: str | None,
    ):
        """Should return a valid IntervalLabel."""
        result = validate_interval_label(
            label=label,
            default=default,
        )
        assert result == label

    @hp.given(default=st.sampled_from(LabelSide))
    def test_should_use_default_when_label_is_none(self, default: LabelSide):
        """Should return the default when label is None."""
        for default in LabelSide:
            result = validate_interval_label(label=None, default=default)
            assert result == default.value


class TestValidateIntervalClosed:
    @hp.given(
        closed=st.one_of(
            st.none(),
            st.just('invalid_closed'),
        ),
        default=st.one_of(
            st.none(),
            st.just('invalid_closed'),
        ),
    )
    def test_should_raise_value_error_for_invalid_arguments(
        self, closed: str | None, default: str | None
    ):
        """Should raise a ValueError for invalid IntervalClosed."""
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
    def test_returns_valid_interval_closed(
        self,
        closed: str | None,
        default: str | None,
    ):
        """Should return a valid IntervalClosed."""
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
    def test_returns_closed_if_closed_is_not_none(
        self,
        closed: str,
        default: str | None,
    ):
        """Should return a valid IntervalClosed."""
        result = validate_interval_closed(
            closed=closed,
            default=default,
        )
        assert result == closed

    @hp.given(default=st.sampled_from(ClosedSide))
    def test_should_use_default_when_closed_is_none(self, default: ClosedSide):
        """Should return the default when closed is None."""
        for default in ClosedSide:
            result = validate_interval_closed(closed=None, default=default)
            assert result == default.value


class TestResolveAxisName:
    @pt.mark.parametrize(
        'key',
        [
            # Coordinate that exists but is not an axis
            'aux',
            # Non-existent coordinate
            'foo',
        ],
    )
    def test_resolve_axis_name_raises_key_error(
        self, ds: xr.Dataset, key: str
    ):
        """Should raise a KeyError if the axis name cannot be resolved."""
        with pt.raises(KeyError):
            resolve_axis_name(ds, key)

    @pt.mark.parametrize(
        'key',
        [
            # CF standard_name
            'longitude',
            # CF axis name
            'X',
            # variable name
            'lon',
        ],
    )
    def test_resolves_axis_name(self, ds: xr.Dataset, key: str):
        """Should resolve the axis name from various CF keys."""
        assert resolve_axis_name(ds, key) == 'X'


class TestResolveStandardName:
    @pt.mark.parametrize('key', ['aux', 'foo'])
    def test_resolve_standard_name_raises_key_error(
        self, ds: xr.Dataset, key: str
    ):
        """Should raise a KeyError if the standard name cannot be resolved."""
        with pt.raises(KeyError):
            resolve_standard_name(ds, key)

    @pt.mark.parametrize(
        'key',
        [
            # CF standard_name
            'latitude',
            # CF axis name
            'Y',
            # variable name
            'lat',
        ],
    )
    def test_resolves_standard_name(self, ds: xr.Dataset, key: str):
        """Should resolve the standard name from various CF keys."""
        assert resolve_standard_name(ds, key) == 'latitude'


class TestResolveVariableName:
    def test_resolve_variable_name_raises_key_error(self, ds: xr.Dataset):
        """Should raise a KeyError if the variable name cannot be resolved."""
        with pt.raises(KeyError):
            resolve_variable_name(ds, 'foo')

    @pt.mark.parametrize(
        'key',
        [
            # CF standard_name
            'latitude',
            # CF axis name
            'Y',
            # variable name
            'lat',
        ],
    )
    def test_resolves_variable_name(self, ds: xr.Dataset, key: str):
        """Should resolve the variable name from various CF keys."""
        assert resolve_variable_name(ds, key) == 'lat'


class TestOffsetAlias:
    @hp.given(freq=xrst.frequencies.offset_aliases())
    def test_from_freq_with_freq_string(self, freq: str):
        """Should return an OffsetAlias object from a frequency string."""
        # Normalize `freq` using pandas to_offset to handle equivalent frequency strings
        # i.e. 1MS == MS, YS-JAN == YS, etc.
        date_offset = pd.tseries.frequencies.to_offset(freq)
        offset_alias = OffsetAlias.from_freq(freq)
        assert isinstance(offset_alias, OffsetAlias)
        assert offset_alias.freqstr == date_offset.freqstr

    @hp.given(freq=xrst.frequencies.offset_aliases())
    def test_from_freq_with_date_offset(self, freq: str):
        """Should return an OffsetAlias object from a pd.DateOffset object."""
        date_offset = pd.tseries.frequencies.to_offset(freq)
        offset_alias = OffsetAlias.from_freq(date_offset)
        assert isinstance(offset_alias, OffsetAlias)
        assert offset_alias.freqstr == date_offset.freqstr

    @hp.given(freq=xrst.frequencies.offset_aliases())
    def test_to_offset_with_freq_string(self, freq: str):
        offset_alias = OffsetAlias.from_freq(freq)
        expected = pd.tseries.frequencies.to_offset(freq)
        actual = offset_alias.offset

        assert expected == actual

    @hp.given(freq=xrst.frequencies.offset_aliases())
    def test_to_offset_with_date_offset(self, freq: str):
        expected = pd.tseries.frequencies.to_offset(freq)
        offset_alias = OffsetAlias.from_freq(expected)
        actual = offset_alias.offset

        assert expected == actual

    @hp.given(
        freq_start=xrst.frequencies.offset_aliases(
            categories=['D', 'MS', 'QS', 'YS']
        ),
        freq_end=xrst.frequencies.offset_aliases(
            categories=['W', 'ME', 'QE', 'YE']
        ),
    )
    def test_is_end_aligned(self, freq_start: str, freq_end: str):
        """Should correctly identify end-aligned frequencies."""
        offset_start = OffsetAlias.from_freq(freq_start)
        offset_end = OffsetAlias.from_freq(freq_end)

        assert offset_start.is_end_aligned is False
        assert offset_end.is_end_aligned is True


class TestMappingOrKwargs:
    @pt.mark.parametrize(
        'pargs, kwargs, func_name, expected',
        [
            ({'a': 1, 'b': 2}, {}, 'test_func', {'a': 1, 'b': 2}),
            (None, {'a': 1, 'b': 2}, 'test_func', {'a': 1, 'b': 2}),
            ({}, {'a': 1, 'b': 2}, 'test_func', {'a': 1, 'b': 2}),
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
        'pargs, kwargs, func_name',
        [
            ({'a': 1}, {'b': 2}, 'test_func'),
        ],
    )
    def test_mapping_or_kwargs_raises_value_error(
        self, pargs, kwargs, func_name
    ):
        with pt.raises(ValueError):
            mapping_or_kwargs(pargs, kwargs, func_name)
