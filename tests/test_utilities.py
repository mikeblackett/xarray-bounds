from collections.abc import Iterable, Mapping

import hypothesis as hp
import hypothesis.extra.numpy as hnp
import hypothesis.extra.pandas as hpd
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest as pt

from tests.constants import FREQ, FREQ_END, FREQ_START
from xarray_bounds.types import ClosedSide, LabelSide
from xarray_bounds.utilities import (
    datetime_to_interval,
    index_to_interval,
    infer_interval,
    infer_midpoint_freq,
)

NUMPY_DTYPES: Mapping[str, st.SearchStrategy[np.dtype]] = {
    'boolean': hnp.boolean_dtypes(),
    'integer': hnp.integer_dtypes(),
    'floating': hnp.floating_dtypes(sizes=(32, 64)),
    'complex': hnp.complex_number_dtypes(),
    'datetime': hnp.datetime64_dtypes(max_period='s'),
}


def index_dtypes(
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> st.SearchStrategy[np.dtype]:
    """Strategy to generate dtypes for indexes."""
    if include and exclude:
        raise ValueError(
            "cannot specify both 'include' and 'exclude' categories."
        )
    if include is not None:
        return st.one_of(*[NUMPY_DTYPES[cat] for cat in include])
    return st.one_of(
        *[
            strat
            for key, strat in NUMPY_DTYPES.items()
            if key not in (exclude or [])
        ]
    )


class TestInferInterval:
    @hp.given(is_datetime=st.booleans())
    def test_delegates_logic(self, is_datetime: bool):
        """Should delegate to the appropriate function based on index type."""
        if is_datetime:
            index = pd.date_range(start='2000-01-01', periods=5, freq='D')
            interval = infer_interval(index=index)
            assert interval.dtype.subtype == 'datetime64[ns]'
        else:
            index = pd.Index(range(5))
            interval = infer_interval(index=index)
            assert interval.dtype.subtype == 'int64'


class TestDatetimeToInterval:
    @hp.given(data=st.data())
    def test_raises_if_index_is_not_datetime(
        self, data: st.DataObject
    ) -> None:
        """Should raise a TypeError if the index is not a DatetimeIndex."""
        dtype = data.draw(index_dtypes(exclude=['datetime']))
        index = data.draw(hpd.indexes(dtype=dtype, max_size=3))
        with pt.raises(TypeError):
            datetime_to_interval(index)  # type: ignore

    @hp.given(data=st.data())
    def test_raises_if_no_inferrable_frequency(self, data: st.DataObject):
        """Should raise a ValueError if the index has no inferrable frequency.

        Could be:
          - irregular;
          - too short;
        """
        label = data.draw(st.sampled_from(LabelSide))
        dtype = data.draw(index_dtypes(include=['datetime']))
        index = data.draw(hpd.indexes(dtype=dtype, max_size=5))
        assert isinstance(index, pd.DatetimeIndex)
        with pt.raises(ValueError):
            datetime_to_interval(index, label=label)

    def test_raises_with_dst_aware_offsets(self):
        """Should raise a ValueError if the index has DST-aware offsets."""
        index = pd.date_range(
            start='2020-03-08', periods=5, freq='3MS', tz='America/New_York'
        )
        with pt.raises(ValueError, match='DST-aware'):
            datetime_to_interval(index)

    @hp.given(
        label=st.sampled_from(['left', 'right']),
        closed=st.sampled_from(ClosedSide),
        start=st.integers(min_value=1890, max_value=2050),
        periods=st.integers(min_value=3, max_value=10),
        freq=st.sampled_from(FREQ),
    )
    def test_converts_regular_indices(
        self,
        start: int,
        periods: int,
        freq: str,
        closed: ClosedSide,
        label: LabelSide,
    ) -> None:
        """Should convert datetime indices with regular frequency to interval indices."""
        expected = pd.interval_range(
            start=pd.Timestamp(str(start)),
            periods=periods,
            freq=freq,
            closed=closed.value,
        )
        index = getattr(expected, label)
        actual = datetime_to_interval(index=index, label=label, closed=closed)
        np.testing.assert_array_equal(actual=actual, desired=expected)

    @hp.given(
        start=st.integers(min_value=1890, max_value=2050),
        periods=st.integers(min_value=4, max_value=10),
        freq=st.sampled_from(FREQ_START),
    )
    def test_converts_start_aligned_midpoint_indices(
        self,
        start: int,
        periods: int,
        freq: str,
    ) -> None:
        """Should convert datetime indices with mid-point frequency to interval indices."""
        closed = ClosedSide('left')
        expected = pd.interval_range(
            start=pd.Timestamp(str(start)),
            periods=periods,
            freq=freq,
            closed=closed.value,
        )
        index = expected.mid
        actual = datetime_to_interval(
            index=index, label=LabelSide.MIDDLE, closed=closed
        )
        np.testing.assert_array_equal(actual=actual, desired=expected)

    @hp.given(
        start=st.integers(min_value=1890, max_value=2050),
        periods=st.integers(min_value=4, max_value=10),
        freq=st.sampled_from(FREQ_END),
    )
    def test_converts_end_aligned_midpoint_indices(
        self,
        start: int,
        periods: int,
        freq: str,
    ) -> None:
        """Should convert datetime indices with mid-point frequency to interval indices."""
        closed = ClosedSide('right')
        expected = pd.interval_range(
            start=pd.Timestamp(str(start)),
            periods=periods,
            freq=freq,
            closed=closed.value,
        )
        index = expected.mid
        actual = datetime_to_interval(
            index=index, label=LabelSide.MIDDLE, closed=closed
        )
        np.testing.assert_array_equal(actual=actual, desired=expected)

    @hp.given(
        name=st.one_of(st.none(), st.text()),
    )
    def test_index_name(self, name: str | None) -> None:
        """Should return interval with the correct name."""
        index = pd.date_range(
            start='2000-01-01', periods=5, freq='D', name='time'
        )
        expected = name or index.name
        interval = datetime_to_interval(index, name=name)
        assert interval.name == expected


class TestInferMidpointFreq:
    @hp.given(data=st.data())
    def test_raises_if_index_is_not_datetime(
        self, data: st.DataObject
    ) -> None:
        """Should raise a TypeError if the index is not a DatetimeIndex."""
        dtype = data.draw(index_dtypes(exclude=['datetime']))
        index = data.draw(hpd.indexes(dtype=dtype))
        with pt.raises(TypeError):
            infer_midpoint_freq(index)  # type: ignore

    @hp.given(data=st.data())
    def test_raises_if_index_is_too_short(self, data: st.DataObject):
        """Should raise a ValueError if the index is too short."""
        dtype = data.draw(index_dtypes(include=['datetime']))
        index = data.draw(hpd.indexes(dtype=dtype, max_size=3))
        with pt.raises(ValueError, match='size'):
            infer_midpoint_freq(index)  # type: ignore

    @hp.given(data=st.data())
    def test_raises_if_index_is_not_monotonic(self, data: st.DataObject):
        """Should raise a ValueError if the index is not monotonic."""
        dtype = data.draw(index_dtypes(include=['datetime']))
        index = data.draw(hpd.indexes(dtype=dtype, min_size=4))
        hp.assume(
            not (
                index.is_monotonic_decreasing or index.is_monotonic_increasing
            )
        )
        with pt.raises(ValueError, match='monotonic'):
            infer_midpoint_freq(index)  # type: ignore

    @hp.given(data=st.data())
    def test_shortcuts_for_daily_frequency(self, data: st.DataObject):
        """Should shortcut for daily frequency."""
        interval = pd.interval_range(
            start=pd.Timestamp('2000-01-01'), periods=5, freq='D'
        )
        index = pd.to_datetime(interval.mid)
        closed = data.draw(st.sampled_from(ClosedSide))
        inferred_freq = infer_midpoint_freq(index=index, closed=closed)
        assert inferred_freq == 'D'

    @hp.given(data=st.data())
    def test_infers_correct_freq(self, data: st.DataObject):
        """Should infer the frequency of a datetime index of regular midpoints"""
        start = data.draw(st.integers(min_value=1890, max_value=2050))
        periods = data.draw(st.integers(min_value=5, max_value=10))
        closed = data.draw(st.sampled_from(ClosedSide))
        freqs = FREQ_START if closed == ClosedSide.LEFT else FREQ_END
        freq = data.draw(st.sampled_from(freqs))

        index = pd.date_range(start=str(start), periods=periods, freq=freq)
        interval = pd.IntervalIndex.from_breaks(index)
        midpoint = pd.to_datetime(interval.mid)

        inferred_freq = infer_midpoint_freq(index=midpoint, closed=closed)  # type: ignore

        # It should be able to infer a frequency string
        assert isinstance(inferred_freq, str)
        try:
            # For most frequencies, the inferred frequency should be equal to
            # the frequency of the original index.
            assert inferred_freq == index.freqstr
        except AssertionError:
            # However, some frequencies are equivalent, but not equal.
            # For example, `3MS` would be inferred as `QS`.
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
        closed=st.sampled_from(ClosedSide),
        label=st.sampled_from(LabelSide),
        name=st.text(),
    )
    def test_raises_if_index_too_short(
        self, closed: ClosedSide, label: LabelSide, name: str
    ):
        """Should raise a ValueError if the index is too short."""
        index = pd.Index([1])
        with pt.raises(ValueError):
            index_to_interval(
                index=index, closed=closed, label=label, name=name
            )

    @hp.given(
        closed=st.sampled_from(ClosedSide),
        label=st.sampled_from(LabelSide),
        name=st.text(),
    )
    def test_raises_if_index_not_monotonic(
        self, closed: ClosedSide, label: LabelSide, name: str
    ):
        """Should raise a ValueError if the index is not monotonic increasing."""
        index = pd.Index([3, 1, 2])
        with pt.raises(ValueError):
            index_to_interval(
                index=index, closed=closed, label=label, name=name
            )

    @hp.given(
        closed=st.sampled_from(ClosedSide),
        label=st.sampled_from(LabelSide),
        name=st.text(),
    )
    def test_raises_if_index_not_uniformly_spaced(
        self, closed: ClosedSide, label: LabelSide, name: str
    ):
        """Should raise a ValueError if the index is not uniformly spaced."""
        index = pd.Index([1, 2, 4, 9])
        with pt.raises(ValueError):
            index_to_interval(
                index=index, closed=closed, label=label, name=name
            )

    @hp.given(
        start=st.integers(min_value=-10, max_value=10),
        periods=st.integers(min_value=3, max_value=10),
        step=st.integers(min_value=1, max_value=3),
        closed=st.sampled_from(ClosedSide),
        label=st.sampled_from(LabelSide),
        name=st.text(),
    )
    def test_roundtrips_increasing_index(
        self,
        start: int,
        periods: int,
        step: int,
        closed: ClosedSide,
        label: LabelSide,
        name: str,
    ):
        """Should be able to roundtrip between interval index and index."""
        expected = pd.interval_range(
            start=start, periods=periods, freq=step, closed=closed.value
        )

        index = getattr(
            expected, 'mid' if label == LabelSide.MIDDLE else label
        )

        actual = index_to_interval(
            index=index, closed=closed, label=label, name=name
        )
        np.testing.assert_array_equal(actual=actual, desired=expected)

    @hp.given(
        start=st.integers(min_value=-10, max_value=10),
        periods=st.integers(min_value=3, max_value=10),
        step=st.integers(min_value=1, max_value=3),
        closed=st.sampled_from(ClosedSide),
        label=st.sampled_from(LabelSide),
        name=st.text(),
    )
    def test_roundtrips_decreasing_index(
        self,
        start: int,
        periods: int,
        step: int,
        closed: ClosedSide,
        label: LabelSide,
        name: str,
    ):
        """Should be able to roundtrip between interval index and index."""
        closed_flipped = (
            ClosedSide.LEFT if closed == ClosedSide.RIGHT else ClosedSide.RIGHT
        )
        label_flipped = (
            LabelSide.LEFT
            if label == LabelSide.RIGHT
            else LabelSide.RIGHT
            if label == LabelSide.LEFT
            else LabelSide.MIDDLE
        )
        expected = pd.interval_range(
            start=start,
            periods=periods,
            freq=step,
            closed=closed_flipped.value,
        )
        expected = pd.IntervalIndex(expected[::-1])

        index = getattr(
            expected,
            'mid' if label_flipped == LabelSide.MIDDLE else label_flipped,
        )

        actual = index_to_interval(
            index=index, closed=closed, label=label, name=name
        )
        np.testing.assert_array_equal(actual=actual, desired=expected)

    @hp.given(name=st.one_of(st.none(), st.text()))
    def test_index_name(self, name: str | None) -> None:
        """Should return interval with the correct name."""
        index = pd.RangeIndex(start=0, stop=5, step=1, name='lat')
        expected = name or index.name
        interval = index_to_interval(index, name=name)
        assert interval.name == expected
