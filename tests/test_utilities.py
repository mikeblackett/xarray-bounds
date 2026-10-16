from collections.abc import Iterable, Mapping
from typing import cast

import hypothesis as hp
import hypothesis.extra.numpy as hnp
import hypothesis.extra.pandas as hpd
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest as pt
import pytest_mock as mk

from tests.constants import FREQ, FREQ_END, FREQ_START
from xarray_bounds.constants import IntervalClosed, IntervalLabel
from xarray_bounds.utilities import (
    _flip_closed,
    _flip_label,
    _left_from_midpoint,
    datetime_to_interval,
    index_to_interval,
    infer_interval,
    infer_midpoint_freq,
)

NUMPY_DTYPES: Mapping[str, st.SearchStrategy[np.dtype]] = {
    "boolean": hnp.boolean_dtypes(),
    "integer": hnp.integer_dtypes(),
    "floating": hnp.floating_dtypes(sizes=(32, 64)),
    "complex": hnp.complex_number_dtypes(),
    "datetime": hnp.datetime64_dtypes(max_period="s"),
    "string": hnp.unicode_string_dtypes(),
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
    @pt.mark.parametrize("is_datetime", [True, False])
    def test_delegates_to_typed_implementation(
        self, mocker: mk.MockerFixture, is_datetime: bool
    ):
        if is_datetime:
            target = "xarray_bounds.utilities.datetime_to_interval"
            index = pd.date_range(start="2000-01-01", periods=5, freq="D")

        else:
            target = "xarray_bounds.utilities.index_to_interval"
            index = pd.Index(range(5))
        mock = mocker.patch(target)

        infer_interval(index=index)
        mock.assert_called_once()


class TestDatetimeToInterval:
    @hp.given(data=st.data())
    def test_other_index_type_raises_type_error(
        self, data: st.DataObject
    ) -> None:
        dtype = data.draw(index_dtypes(exclude=["datetime"]))
        index = data.draw(hpd.indexes(dtype=dtype, max_size=3))
        with pt.raises(TypeError):
            datetime_to_interval(index)  # type: ignore

    @hp.given(periods=st.integers(min_value=0, max_value=2))
    def test_short_index_raises_value_error(self, periods: int):
        # pandas needs >= 3 items to infer freq.
        index = pd.date_range(start="2000", periods=periods)
        assert isinstance(index, pd.DatetimeIndex)
        with pt.raises(ValueError):
            datetime_to_interval(index)

    @hp.given(data=st.data())
    def test_irregular_index_raises_value_error(self, data: st.DataObject):
        dtype = data.draw(index_dtypes(include=["datetime"]))
        index = data.draw(hpd.indexes(dtype=dtype, max_size=5))
        assert isinstance(index, pd.DatetimeIndex)
        with pt.raises(ValueError):
            datetime_to_interval(index)

    @pt.mark.parametrize("freq", ["MS", "3YS", "D"])
    def test_non_utc_index_raises_value_error(self, freq: str):
        index = pd.date_range(
            start="2020-03-08", periods=5, freq=freq, tz="America/New_York"
        )
        with pt.raises(ValueError, match="timezone"):
            datetime_to_interval(index)

    @hp.given(
        label=st.sampled_from(["left", "right"]),
        closed=st.sampled_from(IntervalClosed),
        start=st.integers(min_value=1890, max_value=2050),
        periods=st.integers(min_value=3, max_value=10),
        freq=st.sampled_from(FREQ),
    )
    def test_regularly_spaced_index_is_converted(
        self,
        start: int,
        periods: int,
        freq: str,
        closed: IntervalClosed,
        label: IntervalLabel,
    ) -> None:
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
    def test_start_aligned_midpoint_index_is_converted(
        self,
        start: int,
        periods: int,
        freq: str,
    ) -> None:
        closed = IntervalClosed.LEFT
        expected = pd.interval_range(
            start=pd.Timestamp(str(start)),
            periods=periods,
            freq=freq,
            closed=closed.value,
        )
        index = pd.to_datetime(expected.mid)
        actual = datetime_to_interval(
            index=index, label=IntervalLabel.MIDDLE, closed=closed
        )
        np.testing.assert_array_equal(actual=actual, desired=expected)

    @hp.given(
        start=st.integers(min_value=1890, max_value=2050),
        periods=st.integers(min_value=4, max_value=10),
        freq=st.sampled_from(FREQ_END),
    )
    def test_end_aligned_midpoint_index_is_converted(
        self,
        start: int,
        periods: int,
        freq: str,
    ) -> None:
        closed = IntervalClosed.RIGHT
        expected = pd.interval_range(
            start=pd.Timestamp(str(start)),
            periods=periods,
            freq=freq,
            closed=closed.value,
        )
        index = pd.to_datetime(expected.mid)
        actual = datetime_to_interval(
            index=index, label=IntervalLabel.MIDDLE, closed=closed
        )
        np.testing.assert_array_equal(actual=actual, desired=expected)

    @hp.given(
        name=st.one_of(st.none(), st.text()),
    )
    def test_interval_has_correct_name(self, name: str | None) -> None:
        index = pd.date_range(
            start="2000-01-01", periods=5, freq="D", name="time"
        )
        expected = name or index.name
        interval = datetime_to_interval(index, name=name)
        assert interval.name == expected

    @hp.given(
        start=st.integers(min_value=1890, max_value=2050),
        periods=st.integers(min_value=3, max_value=10),
        freq=st.sampled_from(FREQ),
        closed=st.sampled_from(IntervalClosed),
        label=st.sampled_from([IntervalLabel.LEFT, IntervalLabel.RIGHT]),
    )
    def test_decreasing_regularly_spaced_index_is_converted(
        self,
        start: int,
        periods: int,
        freq: str,
        closed: IntervalClosed,
        label: IntervalLabel,
    ) -> None:
        # A decreasing index should produce the mirror image of the
        # increasing case: the same intervals in reverse order, with the
        # closed side flipped.
        closed_flipped = (
            IntervalClosed.RIGHT
            if closed is IntervalClosed.LEFT
            else IntervalClosed.LEFT
        )
        label_flipped = (
            IntervalLabel.RIGHT
            if label is IntervalLabel.LEFT
            else IntervalLabel.LEFT
        )
        expected = pd.interval_range(
            start=pd.Timestamp(str(start)),
            periods=periods,
            freq=freq,
            closed=closed_flipped.value,
        )
        expected = pd.IntervalIndex(expected.to_list()[::-1])
        index = pd.to_datetime(getattr(expected, label_flipped.value))
        actual = datetime_to_interval(index=index, label=label, closed=closed)
        np.testing.assert_array_equal(actual=actual, desired=expected)

    @hp.given(
        start=st.integers(min_value=1890, max_value=2050),
        periods=st.integers(min_value=4, max_value=10),
        freq=st.sampled_from(FREQ_START),
    )
    def test_decreasing_start_aligned_midpoint_index_is_converted(
        self,
        start: int,
        periods: int,
        freq: str,
    ) -> None:
        closed = IntervalClosed.RIGHT
        expected = pd.interval_range(
            start=pd.Timestamp(str(start)),
            periods=periods,
            freq=freq,
            closed=IntervalClosed.LEFT.value,
        )
        expected = pd.IntervalIndex(expected.to_list()[::-1])
        index = pd.to_datetime(expected.mid)
        actual = datetime_to_interval(
            index=index, label=IntervalLabel.MIDDLE, closed=closed
        )
        np.testing.assert_array_equal(actual=actual, desired=expected)

    @hp.given(
        start=st.integers(min_value=1890, max_value=2050),
        periods=st.integers(min_value=4, max_value=10),
        freq=st.sampled_from(FREQ_END),
    )
    def test_decreasing_end_aligned_midpoint_index_is_converted(
        self,
        start: int,
        periods: int,
        freq: str,
    ) -> None:
        closed = IntervalClosed.LEFT
        expected = pd.interval_range(
            start=pd.Timestamp(str(start)),
            periods=periods,
            freq=freq,
            closed=IntervalClosed.RIGHT.value,
        )
        expected = pd.IntervalIndex(expected.to_list()[::-1])
        index = pd.to_datetime(expected.mid)
        actual = datetime_to_interval(
            index=index, label=IntervalLabel.MIDDLE, closed=closed
        )
        np.testing.assert_array_equal(actual=actual, desired=expected)

    def test_decreasing_midpoint_returns_correct_intervals(self):
        # Regression test: decreasing midpoint indices previously produced
        # duplicated and shifted intervals.
        expected = pd.interval_range(
            start=pd.Timestamp("2000-01-01"),
            periods=4,
            freq="ME",
            closed=IntervalClosed.RIGHT.value,
        )
        expected = pd.IntervalIndex(expected.to_list()[::-1])
        index = pd.to_datetime(expected.mid)
        actual = datetime_to_interval(
            index=index, label=IntervalLabel.MIDDLE, closed=IntervalClosed.LEFT
        )
        np.testing.assert_array_equal(actual=actual, desired=expected)

    @hp.given(
        freq=st.sampled_from(["D", "MS", "ME", "W"]),
        # Only edge labels are used here: for midpoint labels, one of the
        # two mirrored ``closed`` sides is not reliably inferrable (a
        # pre-existing limitation, covered by the midpoint roundtrip tests
        # above).
        label=st.sampled_from([IntervalLabel.LEFT, IntervalLabel.RIGHT]),
        closed=st.sampled_from(IntervalClosed),
        periods=st.integers(min_value=4, max_value=8),
    )
    def test_decreasing_index_is_mirror_of_increasing(
        self,
        freq: str,
        label: IntervalLabel,
        closed: IntervalClosed,
        periods: int,
    ) -> None:
        expected = pd.interval_range(
            start=pd.Timestamp("2000-01-01"),
            periods=periods,
            freq=freq,
            closed=closed.value,
        )
        index = pd.to_datetime(getattr(expected, label.value))
        actual = datetime_to_interval(index=index, label=label, closed=closed)
        mirrored = datetime_to_interval(
            index=index[::-1],
            label=_flip_label(label),
            closed=_flip_closed(closed),
        )
        np.testing.assert_array_equal(actual, mirrored[::-1])


class TestInferMidpointFreq:
    @hp.given(data=st.data())
    def test_other_index_type_raises_type_error(
        self, data: st.DataObject
    ) -> None:
        dtype = data.draw(index_dtypes(exclude=["datetime"]))
        index = data.draw(hpd.indexes(dtype=dtype))
        with pt.raises(TypeError):
            infer_midpoint_freq(index)  # type: ignore

    @hp.given(data=st.data())
    def test_short_index_raises_value_error(self, data: st.DataObject):
        # mid-point inferrence needs 1 more than pandas infer_freq,
        # because we lose 1 when calculating the diffs
        dtype = data.draw(index_dtypes(include=["datetime"]))
        index = data.draw(hpd.indexes(dtype=dtype, max_size=3))
        with pt.raises(ValueError, match="size"):
            infer_midpoint_freq(index)  # type: ignore

    @hp.given(data=st.data())
    def test_non_monotonic_index_raises_value_error(self, data: st.DataObject):
        dtype = data.draw(index_dtypes(include=["datetime"]))
        index = data.draw(hpd.indexes(dtype=dtype, min_size=4))
        hp.assume(
            not (
                index.is_monotonic_decreasing or index.is_monotonic_increasing
            )
        )
        with pt.raises(ValueError, match="monotonic"):
            infer_midpoint_freq(index)  # type: ignore

    def test_daily_index_delegates_to_pandas(self, mocker: mk.MockerFixture):
        # we dont currently support sub-daily frequencies...
        interval = pd.interval_range(
            start=pd.Timestamp("2000-01-01"), periods=5, freq="D"
        )
        index = pd.to_datetime(interval.mid)
        mock = mocker.patch("pandas.infer_freq", return_value="D")
        infer_midpoint_freq(index=index)
        mock.assert_called_once()

    @hp.given(data=st.data())
    def test_frequency_is_inferred(self, data: st.DataObject):
        start = data.draw(st.integers(min_value=1890, max_value=2050))
        periods = data.draw(st.integers(min_value=5, max_value=10))
        closed = data.draw(st.sampled_from(IntervalClosed))
        freqs = FREQ_START if closed == IntervalClosed.LEFT else FREQ_END
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

    def test_decreasing_datetime_index_is_mirrored(self):
        # A decreasing index should be mirrored to the increasing case,
        # with the closed side flipped.
        expected = pd.interval_range(
            start=pd.Timestamp("2000-01-01"), periods=6, freq="ME"
        )
        increasing = pd.to_datetime(expected.mid)
        decreasing = increasing[::-1]
        assert (
            infer_midpoint_freq(increasing, closed=IntervalClosed.RIGHT)
            == "ME"
        )
        assert (
            infer_midpoint_freq(decreasing, closed=IntervalClosed.LEFT) == "ME"
        )


class TestIndexToInterval:
    @hp.given(data=st.data())
    def test_non_numeric_or_boolean_index_raises_type_error(
        self, data: st.DataObject
    ):
        dtype = data.draw(index_dtypes(include=["boolean", "string"]))
        index = data.draw(hpd.indexes(dtype=dtype))
        with pt.raises(TypeError):
            index_to_interval(index)

    @hp.given(
        closed=st.sampled_from(IntervalClosed),
        label=st.sampled_from(IntervalLabel),
        name=st.text(),
    )
    def test_short_index_raises_value_error(
        self, closed: IntervalClosed, label: IntervalLabel, name: str
    ):
        index = pd.Index([1])
        with pt.raises(ValueError):
            index_to_interval(
                index=index, closed=closed, label=label, name=name
            )

    @hp.given(
        closed=st.sampled_from(IntervalClosed),
        label=st.sampled_from(IntervalLabel),
        name=st.text(),
    )
    def test_non_monotonic_index_raises_value_error(
        self, closed: IntervalClosed, label: IntervalLabel, name: str
    ):
        index = pd.Index([3, 1, 2])
        with pt.raises(ValueError):
            index_to_interval(
                index=index, closed=closed, label=label, name=name
            )

    @hp.given(
        closed=st.sampled_from(IntervalClosed),
        label=st.sampled_from(IntervalLabel),
        name=st.text(),
    )
    def test_irregularly_spaced_index_raises_value_error(
        self, closed: IntervalClosed, label: IntervalLabel, name: str
    ):
        index = pd.Index([1, 2, 4, 9])
        with pt.raises(ValueError):
            index_to_interval(
                index=index, closed=closed, label=label, name=name
            )

    @hp.given(
        start=st.integers(min_value=-10, max_value=10),
        periods=st.integers(min_value=3, max_value=10),
        step=st.integers(min_value=1, max_value=3),
        closed=st.sampled_from(IntervalClosed),
        label=st.sampled_from(IntervalLabel),
        name=st.text(),
    )
    def test_increasing_index_roundtrips(
        self,
        start: int,
        periods: int,
        step: int,
        closed: IntervalClosed,
        label: IntervalLabel,
        name: str,
    ):
        expected = pd.interval_range(
            start=start, periods=periods, freq=step, closed=closed.value
        )

        index = getattr(
            expected, "mid" if label == IntervalLabel.MIDDLE else label
        )

        actual = index_to_interval(
            index=index, closed=closed, label=label, name=name
        )
        np.testing.assert_array_equal(actual=actual, desired=expected)

    @hp.given(
        start=st.integers(min_value=-10, max_value=10),
        periods=st.integers(min_value=3, max_value=10),
        step=st.integers(min_value=1, max_value=3),
        closed=st.sampled_from(IntervalClosed),
        label=st.sampled_from(IntervalLabel),
        name=st.text(),
    )
    def test_decreasing_index_roundtrips(
        self,
        start: int,
        periods: int,
        step: int,
        closed: IntervalClosed,
        label: IntervalLabel,
        name: str,
    ):
        closed_flipped = (
            IntervalClosed.LEFT
            if closed == IntervalClosed.RIGHT
            else IntervalClosed.RIGHT
        )
        label_flipped = (
            IntervalLabel.LEFT
            if label == IntervalLabel.RIGHT
            else IntervalLabel.RIGHT
            if label == IntervalLabel.LEFT
            else IntervalLabel.MIDDLE
        )
        expected = pd.interval_range(
            start=start,
            periods=periods,
            freq=step,
            closed=closed_flipped.value,
        )
        expected = pd.IntervalIndex(expected.to_list()[::-1])

        index = getattr(
            expected,
            "mid" if label_flipped == IntervalLabel.MIDDLE else label_flipped,
        )

        actual = index_to_interval(
            index=index, closed=closed, label=label, name=name
        )
        np.testing.assert_array_equal(actual=actual, desired=expected)

    @hp.given(name=st.one_of(st.none(), st.text()))
    def test_interval_has_correct_name(self, name: str | None) -> None:
        index = pd.RangeIndex(start=0, stop=5, step=1, name="lat")
        expected = name or index.name
        interval = index_to_interval(index, name=name)
        assert interval.name == expected


def test_left_from_freq_with_incompatible_freq_warns(recwarn):
    real_offset = cast(pd.DateOffset, pd.tseries.offsets.MonthBegin())
    expected = pd.interval_range(
        start=pd.Timestamp("2000"),
        periods=10,
        freq=real_offset,
        closed="left",
    )
    index = pd.to_datetime(expected.mid)

    test_offset = pd.tseries.offsets.MonthEnd()
    assert isinstance(test_offset, pd.DateOffset)
    with pt.warns():
        _left_from_midpoint(index, test_offset)


def test_left_from_freq_with_compatible_freq_does_not_warn(recwarn):
    offset = pd.tseries.offsets.MonthBegin()
    expected = pd.interval_range(
        start=pd.Timestamp("2000"),
        periods=10,
        freq=offset,
        closed="left",
    )
    index = pd.to_datetime(expected.mid)
    assert isinstance(offset, pd.DateOffset)
    _left_from_midpoint(index, offset)
    assert len(recwarn) == 0
