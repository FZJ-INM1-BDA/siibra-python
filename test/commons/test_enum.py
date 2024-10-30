import pytest
from siibra.commons.enum import ContainedInEnum


class MyEnum(ContainedInEnum):
    FOO = "foo"
    BAR = "bar"


args = [
    (MyEnum.FOO, True),
    (MyEnum.BAR, True),
    ("foo", True),
    ("bar", True),
    ("FOO", False),
    ("BAR", False),
    ("buzz", False),
    (1, False),
    (None, False),
]


@pytest.mark.parametrize("t_val, expected", args)
def test_containedin_enum(t_val, expected):
    assert (t_val in MyEnum) == expected
