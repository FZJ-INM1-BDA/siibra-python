from typing import Generic, TypeVar, Type, Callable, Dict, List
from functools import wraps
from collections import defaultdict

try:
    from typing_extensions import ParamSpec
except ImportError:
    from typing import ParamSpec

from ..cache import fn_call_cache

T = TypeVar("T")
P = ParamSpec("P")


class RegisterRecall(Generic[P]):
    def __init__(self, cache=True) -> None:
        self._registry: Dict[Type, List[Callable[P, List[T]]]] = defaultdict(list)

        self._registered_count: Dict[Type, int] = {}
        if cache:
            # TODO this... doesn't work too well yet
            # e.g. after register brainglobe, it claims cannot pickle
            # in practice, it should ignore the function that was registered
            self._cached_iter = fn_call_cache(
                self._cached_iter,
                cache_validation_callback=self._cache_invalidation_cb,
                ignore=["self"]
            )

    def register(self, _type: Type[T]):
        def outer(fn):

            @wraps(fn)
            def inner(*args, **kwargs):
                return fn(*args, **kwargs)

            self._registry[_type].append(inner)
            return inner

        return outer

    def iter_fn(self, _type: Type[T]):
        return self._registry[_type]

    # n.b. _count act as pseudo cache invalidation
    def _cached_iter(self, _type: Type[T], _count, *args, **kwargs):
        return [item for fn in self._registry[_type] for item in fn(*args, **kwargs)]

    def iter(self, _type: Type[T], *args, **kwargs):
        _count = self._registry[_type]
        return self._cached_iter(_type, _count, *args, **kwargs)

    def _cache_invalidation_cb(self, metadata):
        _type = metadata["input_args"].get("_type")
        _registered_count = len(self._registry[_type])
        if _type not in self._registered_count:
            self._registered_count[_type] = _registered_count
            return True
        return _registered_count == self._registered_count[_type]
