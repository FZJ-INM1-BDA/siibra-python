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


from collections import defaultdict
from typing import Dict, List

from ...commons_new.logger import logger
from ...attributes.descriptions import register_modalities, Modality, RegionSpec, ID, Name
from ...concepts import QueryParam, Feature, QueryParamCollection
from ...assignment.assignment import filter_by_query_param

cell_body_density_modality = Modality(category="cellular",
                                      value="Cell body density")

source_feature_modality = Modality(value="Segmented cell body density")

@register_modalities()
def register_cell_body_density():
    yield cell_body_density_modality


@filter_by_query_param.register(Feature)
def iter_cell_body_density(input: QueryParamCollection):
    from ..configuration import iter_preconf_features
    mods = [mod for cri in input.criteria for mod in cri._find(Modality)]
    if cell_body_density_modality not in mods:
        return

    name_to_regionspec: Dict[str, RegionSpec] = {}
    returned_features: Dict[str, List[Feature]] = defaultdict(list)

    for feature in iter_preconf_features(
        QueryParamCollection(criteria=[
            QueryParam(attributes=[source_feature_modality])])
    ):
        try:
            regionspec = feature._get(RegionSpec)
            returned_features[regionspec.value].append(feature)
            name_to_regionspec[regionspec.value] = regionspec
        except Exception as e:
            logger.warn(f"Processing {feature} resulted in exception {str(e)}")

    for regionname, features in returned_features.items():
        yield Feature(
            attributes=[
                *[
                    attr
                    for feature in features
                    for attr in feature.attributes
                    if not isinstance(attr, (RegionSpec, Name, ID))
                ],
                name_to_regionspec[regionname],
            ]
        )
