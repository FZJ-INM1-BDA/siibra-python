from typing import TypeVar, Iterable, Callable
from ..exceptions import NonUniqueError

T = TypeVar("T")


def get_ooo(input: Iterable[T], filter_fn: Callable[[T], bool]) -> T:
    """Get One and Only One from an iterable

    Parameters
    ----------
    input: Iterable[T]

    Returns
    -------
    T

    Raises
    ------
    NonUniqueError

    """
    result = list(filter(filter_fn, input))
    try:
        assert len(result) == 1
    except AssertionError as e:
        raise NonUniqueError from e
    return result[0]


def assert_ooo(input: Iterable[T]) -> T:
    """Assert One and Only One from an iterable, and return the only element.

    Parameters
    ----------
    input: Iterable[T]

    Returns
    -------
    T

    Raises
    ------
    NonUniqueError
    """
    listed_input = list(input)
    assert len(listed_input) == 1, f"Expected one item, but got {len(input)}"
    return listed_input[0]
