from xml.etree import ElementTree
import requests
from typing import List
import numpy as np
from dataclasses import replace
import pandas as pd
from time import sleep

from ...commons import logger
from ...cache import fn_call_cache
from ...assignment import register_collection_generator
from ...assignment.attribute_match import intersect_ptcld_image
from ...concepts import Feature, AttributeCollection
from ...descriptions import Gene, Modality, register_modalities
from ...locations import PointCloud
from ...dataitems import Image, Tabular
from ...dataitems.tabular import X_DATA
from ...exceptions import ExternalApiException

modality_of_interest = Modality(value="Gene Expressions")

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

MNI152_SPACE_ID = (
    "minds/core/referencespace/v1.0.0/dafcffc5-4826-4bf1-8ff6-46b8a31ff8e2"
)


@register_modalities()
def add_allen_modality():
    yield modality_of_interest


BASE_URL = "http://api.brain-map.org/api/v2/data"


class _AllenGeneQuery:

    _QUERY = {
        "probe": BASE_URL
        + "/query.xml?criteria=model::Probe,rma::criteria,[probe_type$eq'DNA'],products[abbreviation$eq'HumanMA'],gene[acronym$eq'{gene}'],rma::options[only$eq'probes.id']",
        "multiple_gene_probe": BASE_URL
        + "/query.xml?criteria=model::Probe,rma::criteria,[probe_type$eq'DNA'],products[abbreviation$eq'HumanMA'],gene[acronym$in{genes}],rma::options[only$eq'probes.id']&start_row={start_row}&num_rows={num_rows}",
        "specimen": BASE_URL
        + "/Specimen/query.json?criteria=[name$eq'{specimen_id}']&include=alignment3d",
        "microarray": BASE_URL
        + "/query.json?criteria=service::human_microarray_expression[probes$in{probe_ids}][donors$eq{donor_id}]",
        "gene": BASE_URL
        + "/Gene/query.json?criteria=products[abbreviation$eq'HumanMA']&num_rows=all",
        "factors": BASE_URL
        + "/query.json?criteria=model::Donor,rma::criteria,products[id$eq2],rma::include,age,rma::options[only$eq%27donors.id,dono  rs.name,donors.race_only,donors.sex%27]&start_row={start_row}&num_rows={num_rows}",
    }
    _DONOR_IDS = ["15496", "14380", "15697", "9861", "12876", "10021"]
    _SPECIMEN_IDS = [
        "H0351.1015",
        "H0351.1012",
        "H0351.1016",
        "H0351.2001",
        "H0351.1009",
        "H0351.2002",
    ]

    _SAMPLE_MRI = "_sample_mri"

    @staticmethod
    @fn_call_cache
    def _call_allen_api(url: str) -> requests.Response:
        curcuit_breaker = 10
        while True:
            curcuit_breaker -= 1
            if curcuit_breaker < 0:
                raise ExternalApiException
            response = requests.get(url)
            try:
                response.raise_for_status()
            except requests.RequestException:
                logger.debug("http exception retrying after 5 seconds")
                continue

            # When the Allen site is not available, they still send a status code 200.
            if "site unavailable" in response.text.lower():
                logger.debug("site unavailable. retrying after 5 seconds")
                # retry after 5 seconds
                sleep(5)
                continue

            return response

    @staticmethod
    @fn_call_cache
    def _retrieve_probe_ids(genes: List[str]):
        assert isinstance(genes, list)
        if len(genes) == 1:
            logger.debug(f"Retrieving probe ids for gene {genes[0]}")
        else:
            logger.debug(f"Retrieving probe ids for genes {', '.join(genes)}")
        start_row = 0
        num_rows = 50
        probe_ids = []
        while True:
            url = _AllenGeneQuery._QUERY["multiple_gene_probe"].format(
                start_row=start_row,
                num_rows=num_rows,
                genes=",".join([f"'{g}'" for g in genes]),
            )

            response = _AllenGeneQuery._call_allen_api(url)
            root = ElementTree.fromstring(response.text)
            num_probes = int(root.attrib["num_rows"])
            total_probes = int(root.attrib["total_rows"])
            assert len(root) == 1
            probe_ids.extend([int(root[0][i][0].text) for i in range(num_probes)])
            if (start_row + num_rows) >= total_probes:
                break
            # retrieve another page
            start_row += num_rows
        return probe_ids

    @staticmethod
    @fn_call_cache
    def _retrieve_specimen(specimen_id: str):
        """
        Retrieves information about a human specimen.
        """
        url = _AllenGeneQuery._QUERY["specimen"].format(specimen_id=specimen_id)

        resp = _AllenGeneQuery._call_allen_api(url)
        resp_json = resp.json()
        if not resp_json.get("success"):
            raise Exception(
                "Invalid response when retrieving specimen information: {}".format(url)
            )
        # we ask for 1 specimen, so list should have length 1
        assert len(resp_json["msg"]) == 1
        specimen = resp_json["msg"][0]
        T = specimen["alignment3d"]
        specimen["donor2icbm"] = np.array(
            [
                [T["tvr_00"], T["tvr_01"], T["tvr_02"], T["tvr_09"]],
                [T["tvr_03"], T["tvr_04"], T["tvr_05"], T["tvr_10"]],
                [T["tvr_06"], T["tvr_07"], T["tvr_08"], T["tvr_11"]],
            ]
        )
        return specimen

    @staticmethod
    @fn_call_cache
    def _retrieve_factors(start_row=0, num_rows=50, total_rows: int = None):
        return_obj = {}
        while True:
            factors_url = _AllenGeneQuery._QUERY["factors"].format(
                start_row=start_row, num_rows=num_rows
            )

            resp = _AllenGeneQuery._call_allen_api(factors_url)
            response = resp.json()
            for item in response["msg"]:
                return_obj.update(
                    {
                        str(item["id"]): {
                            "race": item["race_only"],
                            "gender": item["sex"],
                            "age": int(item["age"]["days"] / 365),
                        }
                    }
                )
            total_factors = total_rows or int(response["total_rows"])
            if (start_row + num_rows) >= total_factors:
                break
            # retrieve another page
            start_row += num_rows
        return return_obj

    @staticmethod
    def _retrieve_microarray(donor_id: str, probe_ids: str):
        """
        Retrieve microarray data for several probes of a given donor, and
        compute the MRI position of the corresponding tissue block in the ICBM
        152 space to generate a SpatialFeature object for each sample.
        """

        if len(probe_ids) == 0:
            raise RuntimeError("needs at least one probe_ids")

        # query the microarray data for this donor
        url = _AllenGeneQuery._QUERY["microarray"].format(
            probe_ids=",".join([str(id) for id in probe_ids]), donor_id=donor_id
        )
        resp = _AllenGeneQuery._call_allen_api(url)
        response = resp.json()

        if not response["success"]:
            raise Exception(
                "Invalid response when retrieving microarray data: {}".format(url)
            )

        probes, samples = [response["msg"][n] for n in ["probes", "samples"]]

        for i, sample in enumerate(samples):

            donor_id = sample["donor"]["id"]
            donor_name = sample["donor"]["name"]

            for probe in probes:
                yield {
                    "expression_level": float(probe["expression_level"][i]),
                    "z_score": float(probe["z-score"][i]),
                    "gene": probe["gene-symbol"],
                    "sample_index": i,
                    "probe_id": probe["id"],
                    "donor_id": donor_id,
                    "donor_name": donor_name,
                    _AllenGeneQuery._SAMPLE_MRI: sample["sample"]["mri"],
                }


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


@register_collection_generator(Feature)
def query_allen_gene_api(input: AttributeCollection):
    if modality_of_interest not in input.get(Modality):
        return

    genes = list(input.get(Gene))
    if len(genes) == 0:
        logger.error(
            f"{modality_of_interest.value} was queried, but no gene was provided. Returning empty array."
        )
        return

    images = list(input.get(Image))
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
    feature = Feature(attributes=attributes)

    yield from [feature]
