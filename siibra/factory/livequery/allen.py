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
from typing import List
import pandas as pd
import numpy as np

from .base import LiveQuery
from ...atlases import Region
from ...cache import fn_call_cache
from ...commons_new.logger import logger
from ...concepts import Feature
from ...retrieval.api_fetcher.allen import _AllenGeneQuery
from ...attributes.descriptions import register_modalities, Modality, Gene
from ...attributes.locations import PointCloud
from ...attributes.dataitems import Tabular
from ...attributes.dataitems.volume.image import intersect_ptcld_image
from ...attributes.dataitems.tabular import X_DATA

modality_of_interest = Modality(value="Gene Expressions")

MNI152_SPACE_ID = (
    "minds/core/referencespace/v1.0.0/dafcffc5-4826-4bf1-8ff6-46b8a31ff8e2"
)

ALLEN_ATLAS_NOTIFICATION = """
For retrieving microarray data, siibra connects to the web API of
the Allen Brain Atlas (© 2015 Allen Institute for Brain Science),
available from https://brain-map.org/api/index.html. Any use of the
microarray data needs to be in accordance with their terms of use,
as specified at https://alleninstitute.org/legal/terms-use/.
"""

DESCRIPTION = """
Gene expressions extracted from microarray data in the Allen Atlas.
"""


@fn_call_cache
def _retrieve_measurements(gene_names: List[str]):

    probe_ids = _AllenGeneQuery._retrieve_probe_ids(gene_names)
    specimen = {
        spcid: _AllenGeneQuery._retrieve_specimen(spcid)
        for spcid in _AllenGeneQuery._SPECIMEN_IDS
    }
    factors = _AllenGeneQuery._retrieve_factors()

    measurement = []
    for donor_id in _AllenGeneQuery._DONOR_IDS:
        for item in _AllenGeneQuery._retrieve_microarray(donor_id, probe_ids):

            # coordinate conversion to ICBM152 standard space
            sample_mri = item.pop(_AllenGeneQuery._SAMPLE_MRI)
            donor_name = item.get("donor_name")
            icbm_coord = (
                np.matmul(
                    specimen[donor_name]["donor2icbm"],
                    sample_mri + [1],
                )
            ).round(2)

            other_info = {
                "race": factors[donor_id]["race"],
                "gender": factors[donor_id]["gender"],
                "age": factors[donor_id]["age"],
            }

            measurement.append(
                {**item, **other_info, "mni_xyz": icbm_coord[:3].tolist()}
            )
    return measurement


@register_modalities()
def add_allen_modality():
    yield modality_of_interest

class AllenLiveQuery(LiveQuery[Feature], generates=Feature):
    def generate(self):
        
        # If none of the 
        all_mods = [mod for li in self.find_attributes(Modality) for mod in li]
        if modality_of_interest not in all_mods:
            return
        all_genes = [mod for li in self.find_attributes(Gene) for mod in li]
        if len(all_genes) == 0:
            logger.warning(f"AllenLiveQueryError: expecting at least one gene, but got {len(all_genes)}.")
            return
        
        regions = self.find_attribute_collections(Region)
        if len(regions) != 1:
            logger.warning(f"AllenLiveQueryError: expecting one and only one Region, but got {len(regions)}.")
            return
        
        region = regions[0]

        # since we are only interested in map in mni152 space
        images = region.find_regional_maps("mni152")

        if len(images) != 1:
            logger.warning(f"AllenLiveQueryError: expecting one and only one Image, but got {len(regions)}.")

        image = images[0]
        print(ALLEN_ATLAS_NOTIFICATION)

        attributes = [replace(modality_of_interest)]

        retrieved_measurements = _retrieve_measurements([g.value for g in all_genes])
        ptcld = PointCloud(
            space_id=MNI152_SPACE_ID,
            coordinates=[measure["mni_xyz"] for measure in retrieved_measurements],
        )
        intersection = intersect_ptcld_image(ptcloud=ptcld, image=image)
        inside_coord_set = set(tuple(coord) for coord in intersection.coordinates)

        dataframe = pd.DataFrame.from_dict(
            [
                measurement
                for measurement in retrieved_measurements
                if tuple(measurement["mni_xyz"]) in inside_coord_set
            ]
        )
        tabular_data_attr = Tabular(extra={X_DATA: dataframe})
        attributes.append(tabular_data_attr)

        ptcld = PointCloud(space_id=MNI152_SPACE_ID, coordinates=list(inside_coord_set))
        attributes.append(ptcld)

        yield Feature(attributes=attributes)

