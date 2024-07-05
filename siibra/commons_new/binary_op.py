from typing import TypeVar, Generic, Callable, Type, Dict, Tuple

T = TypeVar("T")
V = TypeVar("V")


class BinaryOp(Generic[T, V]):

    def __init__(self):
        self._store_dict: Dict[
            Tuple[Type[T], Type[T]], Tuple[Callable[[T, T], V], bool]
        ] = {}

    def register(self, a: Type[T], b: Type[T]):
        def outer(fn: Callable[[T, T], V]):
            forward_key = a, b
            backward_key = b, a

            assert forward_key not in self._store_dict, f"{forward_key} already exist"
            assert backward_key not in self._store_dict, f"{backward_key} already exist"

            self._store_dict[backward_key] = fn, True
            self._store_dict[forward_key] = fn, False

            return fn

        return outer

    def get(self, a: T, b: T):
        typea = type(a)
        typeb = type(b)
        key = typea, typeb
        return self._store_dict.get(key)
