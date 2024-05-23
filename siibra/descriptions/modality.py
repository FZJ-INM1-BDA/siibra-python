from dataclasses import dataclass
from functools import wraps
from typing import List, Callable, Iterable, Dict

from .base import Description
from ..commons import create_key


@dataclass
class Modality(Description):
    schema = "siibra/attr/desc/modality/v0.1"
    value: str = None


modalities_generator: List[Callable[[], Iterable[Modality]]] = []
def register_modalities():
    def outer(fn):
        modalities_generator.append(fn)
        @wraps(fn)
        def inner(*args, **kwargs):
            yield from fn(*args, **kwargs)
        return inner
    return outer

class _Vocab:

    def __init__(self):
        self._modalities_fetched_flag = False
        self.mapping: Dict[str, Modality] = {}
    
    def _refresh_modalities(self):
        if self._modalities_fetched_flag:
            return
        import siibra.factory
        self._modalities_fetched_flag = True
        for gen in modalities_generator:
            for item in gen():
                key = create_key(item.value)
                if key in self.mapping:
                    continue
                self.mapping[key] = item

    def __dir__(self):
        self._refresh_modalities()
        return list(self.mapping.keys())
    
    def __getattr__(self, key: str):
        self._refresh_modalities()
        if key in self.mapping:
            return self.mapping[key]
        raise AttributeError(f"{key} not found")
    

vocab = _Vocab()
