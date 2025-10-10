from typing import Literal
import hypothesis as hp
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest as pt
import xarray as xr

import xarray_strategies as xrst

from xarray_bounds.helpers import (
    infer_midpoint_freq,
    midpoint_to_interval,
    index_to_interval,
    infer_interval,
    infer_bounds,
    infer_freq,
)


@hp.given(
    start=st.integers(min_value=-10, max_value=10),
    size=st.integers(min_value=3, max_value=10),
    step=st.integers(min_value=1, max_value=3),
    closed=st.sampled_from(['left', 'right']),
)
def test_midpoint_to_interval(
    start: int,
    size: int,
    step: int,
    closed: Literal['left', 'right'],
):
    """Should be able to roundtrip between interval index and midpoint index.

    The values should be the same, but the types may differ (the
    inferred interval will have the dtype of the midpoint index).
    """
    # Arrange
    end = start + size * step
    expected = pd.interval_range(
        start=start, end=end, freq=step, closed=closed
    )
    index = expected.mid
    # Act
    actual = midpoint_to_interval(index=index, closed=closed)
    # Assert
    np.testing.assert_array_equal(actual=actual, desired=expected)


class TestInferFreq:
    """
    Tests for infer_freq function.
    """

    def test_raises_if_index_not_datetime_like(self):
        """
        Should raise a TypeError if the index is not a datetime-like.
        """
        index = pd.Index(range(10))
        with pt.raises(TypeError):
            infer_freq(index)  # type: ignore

    def test_returns_none_for_sub_daily_frequency(self):
        """
        Should return None for sub-daily frequencies.
        """
        index = pd.date_range(start='2000', periods=10, freq='h')
        assert infer_freq(index) is None

    @hp.given(
        index=xrst.indexes.datetime_indexes(min_size=3),
    )
    def test_infers_freq(
        self,
        index: pd.DatetimeIndex,
    ):
        """
        Should infer the frequency of a regular datetime index.

        The inferred freq can be equivalent, but not equal, to the original
        index freq. We can test equivalence by recreating the original index
        using the inferred freq and comparing the two.
        """
        freq = infer_freq(index)
        assert isinstance(freq, str)
        actual = pd.DatetimeIndex(data=index.values, freq=freq)
        np.testing.assert_array_equal(actual=actual, desired=index)

    @hp.given(data=st.data())
    def test_infers_freq_from_midpoint(
        self,
        data: st.DataObject,
    ):
        """
        Should infer the frequency of a regular datetime index of midpoints.
        """
        type_ = data.draw(st.sampled_from(['index', 'data_array']))
        freqs = xrst.frequencies.offset_aliases(
            min_n=1, max_n=3, exclude_categories=['ME', 'QE', 'YE']
        )
        index = data.draw(
            xrst.indexes.datetime_indexes(
                freqs=freqs, min_size=5, normalize=True
            )
        )

        interval = pd.IntervalIndex.from_breaks(index)
        mid = pd.to_datetime(interval.mid)
        # If the index has an inferrable freq, it wouldn't take the
        # `infer_midpoint_freq` path...
        hp.assume(pd.infer_freq(mid) is None)

        if type_ == 'index':
            midpoint = mid
        else:
            midpoint = mid.to_series().to_xarray().rename(index='time')
        freq = infer_freq(midpoint)
        assert isinstance(freq, str)
        actual = pd.DatetimeIndex(data=index.values, freq=freq)
        np.testing.assert_array_equal(actual, index)


class TestInferMidpointFreq:
    """
    Tests for infer_midpoint_freq function.
    """

    def test_raises_if_index_is_not_datetime(self):
        """
        Should raise a TypeError if the index is not a datetime-like.
        """
        index = pd.Index([1, 2, 3])
        with pt.raises(TypeError):
            infer_midpoint_freq(index)  # type: ignore

    @hp.given(index=xrst.indexes.datetime_indexes(max_size=3))
    def test_raises_if_index_is_too_short(self, index: pd.DatetimeIndex):
        """
        Should raise a ValueError if the index is too short.
        """
        with pt.raises(ValueError):
            infer_midpoint_freq(index)

    @hp.given(data=st.data())
    def test_correctly_infers_freq(self, data: st.DataObject):
        """
        Should correctly infer the frequency of a datetime index of midpoints
        """
        how: Literal['start', 'end'] = data.draw(
            st.sampled_from(['start', 'end'])
        )
        exclude_categories = (
            ['ME', 'QE', 'YE'] if how == 'start' else ['MS', 'QS', 'YS']
        )
        freqs = xrst.frequencies.offset_aliases(
            min_n=1,
            max_n=3,
            exclude_categories=exclude_categories,  # type: ignore
        )
        index = data.draw(
            xrst.indexes.datetime_indexes(
                freqs=freqs,
                min_size=5,
                normalize=True,
            )
        )
        interval = pd.IntervalIndex.from_breaks(index)
        midpoint = pd.to_datetime(interval.mid)

        if data.draw(st.booleans()):
            midpoint = midpoint.to_series().to_xarray().rename(index='time')

        inferred_freq = infer_midpoint_freq(obj=midpoint, how=how)

        # It should be able to infer a frequency string
        assert isinstance(inferred_freq, str)
        try:
            # For most frequencies, the inferred frequency should be equal to
            # the frequency of the original index.
            assert inferred_freq == index.freqstr
        except AssertionError:
            # However, some frequencies are equivalent, but not equal.
            # For example, 3MS would be inferred as QS.
            # In this case we can test equivalence by recreating the
            # original index using the inferred frequency.
            actual = pd.date_range(
                start=index[0],
                periods=index.size,
                freq=inferred_freq,
            )
            np.testing.assert_array_equal(actual, index)


class TestIndexToInterval:
    @hp.given(
        start=st.integers(min_value=-10, max_value=10),
        size=st.integers(min_value=3, max_value=10),
        step=st.integers(min_value=1, max_value=3),
        closed=st.sampled_from(['left', 'right']),
        label=st.sampled_from(['left', 'right']),
    )
    def test_roundtrip(
        self,
        start: int,
        size: int,
        step: int,
        closed: Literal['left', 'right'],
        label: Literal['left', 'right'],
    ):
        """Should be able to roundtrip between interval index and index."""
        # Arrange
        end = start + size * step
        expected = pd.interval_range(
            start=start, end=end, freq=step, closed=closed
        )
        index = getattr(expected, label)
        # Act
        actual = index_to_interval(index=index, closed=closed, label=label)
        # Assert
        np.testing.assert_array_equal(actual=actual, desired=expected)

    @hp.given(
        start=st.integers(min_value=-10, max_value=10),
        size=st.integers(min_value=3, max_value=10),
        step=st.integers(min_value=1, max_value=3),
    )
    def test_delegates_to_midpoint(
        self,
        start: int,
        size: int,
        step: int,
    ):
        """Should delegate to midpoint_to_interval if the label is `middle`."""
        # Arrange
        index = pd.Index(range(start, start + size * step, step))
        expected = midpoint_to_interval(index=index)
        # Act
        actual = index_to_interval(index=index, label='middle')
        # Assert
        np.testing.assert_array_equal(actual=actual, desired=expected)


class TestInferInterval:
    def test_infers_from_datetime_index(self):
        """Should infer the interval from a DatetimeIndex."""
        # Arrange
        index = pd.date_range(start='2000-01-01', periods=10, freq='MS')
        start = index[0]
        assert index.freq is not None
        end = index[-1] + index.freq
        expected = pd.interval_range(
            start=start, end=end, freq=index.freqstr, closed='left'
        )
        # Act
        actual = infer_interval(index=index)
        # Assert
        np.testing.assert_array_equal(actual=actual, desired=expected)

    def test_falls_back_to_midpoint(self):
        """Should fall back to midpoint_to_interval if the datetime index has no frequency."""
        # Arrange
        index = pd.DatetimeIndex(['2000-01-01', '2000-02-01', '2000-04-01'])
        expected = midpoint_to_interval(index=index)
        # Act
        actual = infer_interval(index=index)
        # Assert
        np.testing.assert_array_equal(actual=actual, desired=expected)

    def test_infers_from_regular_index(self):
        """infer the interval from a regular index."""
        # Arrange
        start = 1
        end = 10
        closed = 'left'
        index = pd.Index(range(start, end))
        expected = pd.interval_range(
            start=start, end=end, freq=1, closed=closed
        )
        # Act
        actual = infer_interval(index=index, closed=closed)
        # Assert
        np.testing.assert_array_equal(actual=actual, desired=expected)


class TestInferBounds:
    def test_raises_if_not_1d(self):
        """Should raise an error if the dimension is not 1D."""

        da = xr.tutorial.open_dataset('air_temperature').air
        with pt.raises(
            ValueError,
            match='Bounds are currently only supported for 1D coordinates.',
        ):
            infer_bounds(da)

    @hp.given(
        closed=st.sampled_from(['left', 'right']),
    )
    def test_infers_bounds(self, closed: Literal['left', 'right']):
        """Should infer the bounds for a 1D coordinate.

        This is not heavily parameterized, as the bounds inference is
        tested in the other tests.
        """
        # Arrange
        da = xr.DataArray(
            data=[1, 2, 3], dims=['time'], coords={'time': [1, 2, 3]}
        )
        expected = xr.DataArray(
            name='time_bounds',
            data=[[1, 2], [2, 3], [3, 4]],
            dims=['time', 'bounds'],
            coords={'time': da.time.assign_attrs(bounds='time_bounds')},
            attrs={
                'closed': closed,
            },
        )
        # Act
        actual = infer_bounds(da, closed=closed)
        # Assert
        xr.testing.assert_identical(actual, expected)
