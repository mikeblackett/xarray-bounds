from enum import Enum, StrEnum
from typing import Literal, TypeAlias

__all__ = [
    "ClosedSide",
    "IntervalClosed",
    "IntervalLabel",
    "LabelSide",
    "is_enum_member",
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


def is_enum_member(enum_cls: type[Enum], value: object) -> bool:
    """Return True if ``value`` is a member of ``enum_cls``.

    A value is considered a member if it is a member instance or is equal
    to the value of a member, mirroring ``Enum.__contains__`` on Python
    3.12 and later.

    Note:
        Python 3.11's ``Enum.__contains__`` raises ``TypeError`` for
        non-member values, so membership is checked against member values
        instead. Once support for Python 3.11 is dropped, replace the body
        of this function with ``return value in enum_cls``.
    """
    if isinstance(value, enum_cls):
        return True
    return value in {member.value for member in enum_cls}
