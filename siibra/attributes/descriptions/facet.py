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

from dataclasses import dataclass

from .base import Description


@dataclass
class Facet(Description):
    schema = "siibra/attr/desc/facet/v0.1"
    key: str = None
    value: str = None

    def __str__(self) -> str:
        return f"{self.key}={self.value}"

    def _iter_zippable(self):
        yield from super()._iter_zippable()
        yield f"Facet: {str(self)}", None, None
