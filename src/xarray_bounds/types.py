from enum import StrEnum
from typing import Literal, TypeAlias

__all__ = [
    "ClosedSide",
    "IntervalClosed",
    "IntervalLabel",
    "LabelSide",
]

IntervalLabel: TypeAlias = Literal["left", "middle", "right"]
"""Options for interval labels. Values are members of the LabelSide enum."""

IntervalClosed: TypeAlias = Literal["left", "right"]
"""Options for interval closed. Values are members of the ClosedSide enum."""


class LabelSide(StrEnum):
    """Options for interval labels"""

    LEFT = "left"
    MIDDLE = "middle"
    RIGHT = "right"


class ClosedSide(StrEnum):
    """Options for interval closed"""

    LEFT = "left"
    RIGHT = "right"
