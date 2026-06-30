import pytest

from typing import List
from json import JSONDecodeError
from requests import RequestException
from functools import wraps

from siibra.retrieval.requests import SiibraHttpRequestError
from siibra.livequeries.allen import InvalidAllenAPIResponseException


def check_duplicate(list_of_str: List[str], fn=lambda a: a):
    seen = set()
    duplicate = set()
    for item in list_of_str:
        if fn(item) in seen:
            duplicate.add(item)
        seen.add(fn(item))
    return duplicate


ALLEN_UNAVAILABLE_EXIT_CODE = 5


def is_allen_api_unavailable_exception(exc: BaseException) -> bool:
    if isinstance(exc, JSONDecodeError):
        return True

    if isinstance(exc, RuntimeError):
        return str(exc) == "Allen institute site unavailable - please try again later."

    if isinstance(exc, InvalidAllenAPIResponseException):
        return True

    if isinstance(
        exc,
        (
            RequestException,
            SiibraHttpRequestError,
        ),
    ):
        return True

    return False


def xfail_if_allen_api_unavailable(test_func):
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        try:
            return test_func(*args, **kwargs)
        except Exception as e:
            if is_allen_api_unavailable_exception(e):
                pytest.xfail(
                    f"Skipping test {test_func.__name__} because the Allen API is unavailable or returned an invalid response:\n{e}"
                )
            raise e

    return wrapper
