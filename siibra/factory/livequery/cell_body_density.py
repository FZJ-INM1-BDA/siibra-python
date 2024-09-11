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


from dataclasses import replace
import pandas as pd
import numpy as np
from typing import Iterator, List

from .base import LiveQuery
from ...attributes.descriptions import (
    register_modalities,
    Modality,
    RegionSpec,
    ID,
    Name,
)
from ...attributes.dataproviders.tabular import TabularDataProvider
from ...operations.base import Merge, DataOp
from ...concepts import Feature
from ...operations.tabular import (
    ConcatTabulars,
    TabularMeanStd,
)

cell_body_density_modality = Modality(category="cellular", value="Cell body density")

source_feature_modality = Modality(value="Segmented cell body density")


@register_modalities()
def register_cell_body_density():
    yield cell_body_density_modality


# TODO just too much custom code. Escape hatch
class ProcessCellBodyDensity(DataOp):
    input: List[pd.DataFrame]
    output: pd.Series
    type = "adhoc/cellbodydensity"
    desc = "Processing Cell Body Density data"

    def run(self, input, **kwargs):
        assert isinstance(input, list)
        assert len(input) == 2
        cells, layers = input
        assert isinstance(cells, pd.DataFrame)
        assert isinstance(layers, pd.DataFrame)

        cells = cells.astype({"layer": int, "label": int})

        counts = cells["layer"].value_counts()
        areas = layers["Area(micron**2)"]
        indices = np.intersect1d(areas.index, counts.index)

        return counts[indices] / areas * 100**2 * 5


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

        providers: List[TabularDataProvider] = []
        for feat in wanted_features:
            layerinfo = [
                replace(tab)
                for tab in feat._find(TabularDataProvider)
                if "layerinfo.txt" in tab.url
            ]
            segments = [
                replace(tab)
                for tab in feat._find(TabularDataProvider)
                if "segments.txt" in tab.url
            ]
            assert (
                len(layerinfo) == 1
            ), f"Expected one and only one layerinfo.txt, but got {len(layerinfo)}"
            assert (
                len(segments) == 1
            ), f"Expected one and only one segments.txt, but got {len(segments)}"

            layerinfo = layerinfo[0]
            segments = segments[0]

            provider = TabularDataProvider(
                retrieval_ops=[
                    # n.b. position matters!
                    Merge.spec_from_dataproviders([segments, layerinfo]),
                    ProcessCellBodyDensity.generate_specs(),
                ]
            )
            providers.append(provider)

        summed_table = TabularDataProvider(
            retrieval_ops=[
                Merge.spec_from_dataproviders(providers),
                ConcatTabulars.generate_specs(axis=1),
            ],
            transformation_ops=[
                TabularMeanStd.generate_specs(index=layerinfo.get_data()["Name"])
            ],
            plot_options={
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
