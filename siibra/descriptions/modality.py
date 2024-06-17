from dataclasses import dataclass
from typing import List, Callable, Iterable, Dict

from .base import Description
from ..commons_new.instance_table import JitInstanceTable
from ..commons import logger


@dataclass
class Modality(Description):
    schema = "siibra/attr/desc/modality/v0.1"
    value: str = None

    def __hash__(self) -> int:
        return hash(self.value)


modalities_generator: List[Callable[[], Iterable[Modality]]] = []
_cached_modality = (
    {}
)  # TODO ugly way to deal with JitInstanceTable calling __getattr__ during autocomplete


def get_modalities() -> Dict[str, Modality]:
    _l = len(modalities_generator)
    if _l in _cached_modality:
        return _cached_modality[_l]
    result = {}
    for mod_fn in modalities_generator:
        try:
            for mod in mod_fn():
                result[mod.value] = mod
        except Exception as e:
            logger.warning(f"Generating modality exception: {str(e)}")

    _cached_modality[_l] = result
    return result


vocab = JitInstanceTable(getitem=get_modalities)


def register_modalities():
    def outer(fn):

        if fn in modalities_generator:
            raise RuntimeError("fn already registered")
        modalities_generator.append(fn)
        return fn

    return outer
