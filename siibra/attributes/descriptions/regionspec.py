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
from ...cache import fn_call_cache


@fn_call_cache
def _decode(regionspec: "RegionSpec"):
    # n.b. method cannot be directly decorated with joblib memory, because of the "self" first argument
    # TODO try to not cache region (or parcellation)
    # it caches the entire tree
    from ...atlases import Region
    from ...concepts import QueryParam
    from ...assignment import find

    return find([QueryParam(attributes=[regionspec])], Region)


@dataclass
class RegionSpec(Description):
    schema = "siibra/attr/desc/regionspec/v0.1"
    parcellation_id: str = None

    def __hash__(self) -> int:
        return hash(f"{self.parcellation_id}{self.value}")

    def decode(self):
        return _decode(self)
