from dataclasses import replace
from typing import List
import pandas as pd
import numpy as np

from ...cache import fn_call_cache
from ...commons import logger
from ...concepts import Feature, QueryParam
from ...retrieval_new.api_fetcher.allen import _AllenGeneQuery
from ...descriptions import register_modalities, Modality, Gene
from ...assignment import filter_by_query_param
from ...locations import PointCloud
from ...dataitems import Image, Tabular
from ...dataitems.volume.image import intersect_ptcld_image
from ...dataitems.tabular import X_DATA

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


@filter_by_query_param.register(Feature)
def query_allen_gene_api(input: QueryParam):
    if modality_of_interest not in input._find(Modality):
        return

    genes = input._find(Gene)
    if len(genes) == 0:
        logger.error(
            f"{modality_of_interest.value} was queried, but no gene was provided. Returning empty array."
        )
        return

    images = input._find(Image)
    if len(images) == 0:
        logger.error(
            f"{modality_of_interest.value} was queried, but input contains no image. Returning empty array."
        )
        return

    if len(images) > 1:
        logger.warning(
            f"{modality_of_interest.value} was queried, but input contains multiple images. First one was selected."
        )

    image = images[0]

    print(ALLEN_ATLAS_NOTIFICATION)

    attributes = [replace(modality_of_interest)]

    retrieved_measurements = _retrieve_measurements([g.value for g in genes])
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
