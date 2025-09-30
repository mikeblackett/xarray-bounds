from typing import Literal
import hypothesis as hp
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest as pt
import xarray as xr

from xarray_bounds.helpers import (
    midpoint_to_interval,
    index_to_interval,
    infer_interval,
    infer_bounds,
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
