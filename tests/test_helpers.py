import hypothesis as hp
import hypothesis.strategies as st
import pandas as pd
import pytest as pt
import xarray as xr
import xarray_strategies as xrst

from xarray_bounds.helpers import (
    OffsetAlias,
    mapping_or_kwargs,
    resolve_dim_name,
)


class TestResolveDimName:
    @hp.given(
        key=st.sampled_from(
            [
                # cf axis
                'Y',
                # name
                'lat',
                # standard_name
                'latitude',
            ]
        )
    )
    def test_resolve_dim(self, key: str):
        """Should return the dimension name if it exists."""
        ds = xr.tutorial.load_dataset('air_temperature')
        assert resolve_dim_name(ds, key) == 'lat'

    @hp.given(key=st.text())
    def test_resolve_dim_raises_key_error(self, key: str):
        """Should raise a KeyError if the dimension cannot be resolved."""
        with pt.raises(KeyError):
            resolve_dim_name(xr.DataArray(), key)


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
        actual = offset_alias.to_offset()

        assert expected == actual

    @hp.given(freq=xrst.frequencies.offset_aliases())
    def test_to_offset_with_date_offset(self, freq: str):
        expected = pd.tseries.frequencies.to_offset(freq)
        offset_alias = OffsetAlias.from_freq(expected)
        actual = offset_alias.to_offset()

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
