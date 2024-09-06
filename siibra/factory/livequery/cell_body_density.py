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
from typing import Dict, Iterator, List

from .base import LiveQuery
from ...commons.logger import logger
from ...attributes.descriptions import (
    register_modalities,
    Modality,
    RegionSpec,
    ID,
    Name,
)
from ...attributes.dataproviders.tabular import TabularDataProvider
from ...operations.base import Merge
from ...concepts import Feature
from ...operations.tabular import ConcatTabulars, GroupByTabular

cell_body_density_modality = Modality(category="cellular", value="Cell body density")

source_feature_modality = Modality(value="Segmented cell body density")


@register_modalities()
def register_cell_body_density():
    yield cell_body_density_modality


class CellbodyDensityAggregator(LiveQuery, generates=Feature):
    def generate(self) -> Iterator:
        from ...factory.configuration import iter_preconfigured_ac
        from ...atlases import Region
        from ...assignment.assignment import match as ac_match
        from ...attributes.attribute_collection import AttributeCollection

        mods = [mod for mods in self.find_attributes(Modality) for mod in mods]

        if cell_body_density_modality not in mods:
            return

        regions = self.find_attribute_collections(Region)
        if len(regions) == 0:
            return

        requested_region_specs = [
            RegionSpec(parcellation_id=r.parcellation.ID, value=r.name) for r in regions
        ]

        wanted_features: List[Feature] = []
        for feature in iter_preconfigured_ac(Feature):
            if all(
                mod.value != source_feature_modality.value
                for mod in feature._find(Modality)
            ):
                continue
            if not ac_match(
                AttributeCollection(attributes=requested_region_specs),
                AttributeCollection(attributes=feature._find(RegionSpec)),
            ):
                continue

            # TODO need to merge *all* subfeatures's layerinfo and calc mean/std
            # N.B. use merge, maybe tabular ops
            wanted_features.append(feature)

        if len(wanted_features) == 0:
            return

        summed_table = TabularDataProvider(
            retrieval_ops=[
                Merge.spec_from_dataproviders(
                    [
                        tab
                        for feat in wanted_features
                        for tab in feat._find(TabularDataProvider)
                        # TODO this is somewhat messy, see if this can be fixed in future
                        if "layerinfo.txt" in tab.url
                    ]
                ),
                ConcatTabulars.generate_specs(),
                GroupByTabular.generate_specs(by="Name"),
            ],
            plot_options={
                "sub_dataframe": ["Area(micron**2)"],
                "kind": "bar",
                "y": "mean",
                "yerr": "std",
                "legend": False,
            },
        )

        yield Feature(
            attributes=[
                cell_body_density_modality,
                source_feature_modality,
                *[
                    RegionSpec(parcellation_id=r.parcellation.ID, value=r.name)
                    for r in regions
                ],
                *[
                    attr
                    for feature in wanted_features
                    for attr in feature.attributes
                    if not isinstance(
                        attr, (RegionSpec, Name, ID, Modality, TabularDataProvider)
                    )
                ],
                summed_table,
            ]
        )
