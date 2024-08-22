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

from typing import Dict, Union, List
from dataclasses import dataclass, field, replace

try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict

from .base import Description


class Target(TypedDict):
    target: str


VolumeRegionMapping = TypedDict(
    "VolumeRegionMapping",
    {
        "target": Union[str, None],
        "@type": Literal["volume/ref"],
        "label": int,
        "color": str,
        "range": List[float],
        "subspace": List[Union[Literal[":"], int]],
    },
)

RowIndexRegionMapping = TypedDict(
    "RowIndexRegionMapping",
    {"target": Union[str, None], "@type": Literal["csv/row-index"], "index": int},
)

RegionMapping = Union[VolumeRegionMapping, RowIndexRegionMapping]


@dataclass
class AttributeMapping(Description):
    schema = "siibra/attr/desc/attribute_mapping/v0.1"

    parcellation_id: str = None
    region_mapping: Dict[str, List[RegionMapping]] = field(default_factory=dict)

    ref_type: Union[
        None,
        Literal[
            "openminds/DatasetVersion",
            "openminds/Dataset",
            "openminds/AtlasAnnotation",
            "minds/core/dataset/v1.0.0",
        ],
    ] = None
    refs: Dict[str, List[Target]] = field(default_factory=dict)

    def filter_by_target(self, target_name: Union[str, None] = None):
        def predicate(target: Target):
            return target.get("target") is None or target["target"] == target_name

        return replace(
            self,
            region_mapping={
                key: [
                    region_mapping
                    for region_mapping in region_mappings
                    if predicate(region_mapping)
                ]
                for key, region_mappings in self.region_mapping.items()
                if any(predicate(region_mapping) for region_mapping in region_mappings)
            },
            refs={
                key: [ref for ref in refs if predicate(ref)]
                for key, refs in self.refs.items()
                if any(predicate(ref) for ref in refs)
            },
        )
