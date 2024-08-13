# Copyright 2018-2024
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Callable, Iterable, Dict
from dataclasses import dataclass
from functools import partial

from .base import Description
from ...commons.instance_table import JitInstanceTable
from ...commons.logger import logger
from ...cache import fn_call_cache

from ..._version import __version__


@dataclass
class Modality(Description):
    schema = "siibra/attr/desc/modality/v0.1"
    category: str = None

    def __hash__(self) -> int:
        return hash((self.category or "") + self.value)

    def _iter_zippable(self):
        yield from super()._iter_zippable()
        desc_text = f"Modality: {self.value}"
        if self.category:
            desc_text += f"(category: {self.category})"
        yield desc_text, None, None


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


def register_modalities():
    def outer(fn):
        if fn in modalities_generator:
            raise RuntimeError("fn already registered")
        modalities_generator.append(fn)
        return fn

    return outer


class ModalityVocab:
    """Class that exist purely for the ease of access of category/modality InstanceTable."""

    category = JitInstanceTable(
        getitem=lambda: {
            value.category: JitInstanceTable(
                getitem=partial(
                    get_modalities,
                    __version__,
                    len(modalities_generator),
                    category=value.category,
                )
            )
            for value in get_modalities(__version__, len(modalities_generator)).values()
            if value.category
        }
    )
    modality = JitInstanceTable(
        getitem=lambda: get_modalities(__version__, len(modalities_generator))
    )


modality_vocab = ModalityVocab()
