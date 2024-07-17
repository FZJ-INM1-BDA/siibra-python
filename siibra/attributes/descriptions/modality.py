from typing import List, Callable, Iterable, Dict
from dataclasses import dataclass
from functools import partial

from .base import Description
from ...commons_new.instance_table import JitInstanceTable
from ...commons_new.logger import logger
from ...commons import __version__
from ...cache import fn_call_cache

@dataclass
class Modality(Description):
    schema = "siibra/attr/desc/modality/v0.1"
    category: str=None

    def __hash__(self) -> int:
        return hash((self.category or "") + self.value)


modalities_generator: List[Callable[[], Iterable[Modality]]] = []

@fn_call_cache
def get_modalities(_version, _len, category=None) -> Dict[str, Modality]:
    # getting all modalities require iterating over all feature iterators. 
    # this can be expensive
    # As a result, the _version and _len of modality generate act as cache busters
    # on disk cache will be busted if:
    # - new version of siibra is used
    # - a new feature hook is added
    # TODO as different feature hook may be added, this may not be accurate
    result = {}
    for mod_fn in modalities_generator:
        try:
            for mod in mod_fn():
                if not category:
                    result[mod.value] = mod
                    continue

                if mod.category == category:
                    result[mod.value] = mod

        except Exception as e:
            logger.warning(f"Generating modality exception: {str(e)}")
    return result


vocab = JitInstanceTable(getitem=lambda: get_modalities(__version__, len(modalities_generator)))
categories = JitInstanceTable(getitem=lambda: {
    value.category: JitInstanceTable(getitem=partial(get_modalities, __version__, len(modalities_generator), category=value.category))
    for value in get_modalities(__version__, len(modalities_generator)).values()
    if value.category })

def register_modalities():
    def outer(fn):

        if fn in modalities_generator:
            raise RuntimeError("fn already registered")
        modalities_generator.append(fn)
        return fn

    return outer
