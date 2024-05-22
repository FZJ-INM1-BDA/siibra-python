from dataclasses import dataclass
from functools import wraps
from typing import List, Callable, Iterable, Dict

from .base import Description
from ..commons import create_key


@dataclass
class Modality(Description):
    schema = "siibra/attr/desc/modality/v0.1"
    value: str = None

    # @staticmethod
    # def _GetAll():
    #     from ...configuration import Configuration
    #     from ..feature import DataFeature

    #     cfg = Configuration()
    #     return {
    #         attr.name
    #         for _, s in cfg.specs.get("siibra/feature/v0.2")
    #         for attr in DataFeature(**s).attributes
    #         if isinstance(attr, ModalityAttribute)
    #     }

    # def matches(self, *args, modality=None, **kwargs):
    #     if isinstance(modality, str) and all_words_in_string(modality, self.name):
    #         return True
    #     return super().matches(*args, modality=modality, **kwargs)


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
        if key in self.mapping:
            return self.mapping[key]
        raise AttributeError(f"{key} not found")
    

vocab = _Vocab()
