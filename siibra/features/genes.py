# Copyright 2018-2021
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

from typing import Dict, List
from .feature import SpatialFeature
from .query import FeatureQuery

from ..commons import logger, Registry
from ..core.space import Space, Point
from ..retrieval import HttpRequest

from os import path
from xml.etree import ElementTree
import numpy as np
import json
import siibra
try:
    # python 3.8+
    from typing import TypedDict
except ImportError:
    # python 3.7 and below
    from typing_extensions import TypedDict


BASE_URL = "http://api.brain-map.org/api/v2/data"

class DonorDict(TypedDict):
    id: int
    name: str
    race: str
    age: int
    gender: str

class SampleStructure(TypedDict):
    id: int
    name: str
    abbreviation: str
    color: str

class GeneExpression(SpatialFeature):
    """
    A spatial feature type for gene expressions.
    """

    def __init__(
        self,
        gene: str,
        location: Point,
        expression_levels: List[float],
        z_scores: List[float],
        probe_ids: List[int],
        donor_info: DonorDict,
        mri_coord: List[int]=None,
        structure: SampleStructure=None,
        top_level_structure: SampleStructure=None,
    ):
        """
        Construct the spatial feature for gene expressions measured in a sample.

        Parameters:
        -----------
        gene : str
            Name of gene
        location : Point
            3D location of the gene expression sample
        expression_levels : list of float
            expression levels measured in possibly multiple probes of the same sample
        z_scores : list of float
            z scores measured in possibly multiple probes of the same sample
        probe_ids : list of int
            The probe_ids corresponding to each z_score element
        donor_info : dict (keys: age, race, gender, donor, speciment)
            Dictionary of donor attributes
        mri_coord : tuple  (optional)
            coordinates in original mri space
        """
        SpatialFeature.__init__(self, location=location)
        self.expression_levels = expression_levels
        self.z_scores = z_scores
        self.donor_info = donor_info
        self.gene = gene
        self.probe_ids = probe_ids
        self.mri_coord = mri_coord
        self.structure = structure
        self.top_level_structure = top_level_structure

    def __repr__(self):
        return " ".join(
            [
                "At (" + ",".join("{:4.0f}".format(v) for v in self.location) + ")",
                " ".join(
                    [
                        "{:>7.7}:{:7.7}".format(k, str(v))
                        for k, v in self.donor_info.items()
                    ]
                ),
                "Expression: [" + ",".join(["%4.1f" % v for v in self.expression_levels]) + "]",
                "Z-score: [" + ",".join(["%4.1f" % v for v in self.z_scores]) + "]",
            ]
        )


# provide a registry of gene names
genename_file = path.join(path.dirname(siibra.__file__), "features", "gene_names.json")
with open(genename_file, "r") as f:
    _genes = json.load(f)
GENE_NAMES = Registry()
list(map(lambda k: GENE_NAMES.add(k, k), _genes.keys()))


class AllenBrainAtlasQuery(FeatureQuery):
    """
    Interface to Allen Human Brain Atlas microarray data.

    This class connects to the web API of the Allen Brain Atlas:
    © 2015 Allen Institute for Brain Science. Allen Brain Atlas API.
    Available from: brain-map.org/api/index.html
    Any use of the data needs to be in accordance with their terms of use, see
    https://alleninstitute.org/legal/terms-use/

    - We have samples from 6 different human donors.
    - Each donor corresponds to exactly 1 specimen (tissue used for study)
    - Each sample was subject to multiple (in fact 4) different probes.
    - The probe data structures contain the list of gene expression of a
      particular gene measured in each sample. Therefore the length of the gene
      expression list in a probe coresponds to the number of samples taken in
      the corresponding donor for the given gene.
    """

    _FEATURETYPE = GeneExpression

    ALLEN_ATLAS_NOTIFICATION = """
    For retrieving microarray data, siibra connects to the web API of
    the Allen Brain Atlas (© 2015 Allen Institute for Brain Science),
    available from https://brain-map.org/api/index.html. Any use of the
    microarray data needs to be in accordance with their terms of use,
    as specified at https://alleninstitute.org/legal/terms-use/.
    """
    _notification_shown = False

    _QUERY = {
        "probe": BASE_URL + "/query.xml?criteria=model::Probe,rma::criteria,[probe_type$eq'DNA'],products[abbreviation$eq'HumanMA'],gene[acronym$eq{gene}],rma::options[only$eq'probes.id']",
        "specimen": BASE_URL + "/Specimen/query.json?criteria=[name$eq'{specimen_id}']&include=alignment3d",
        "microarray": BASE_URL + "/query.json?criteria=service::human_microarray_expression[probes$in{probe_ids}][donors$eq{donor_id}]",
        "gene": BASE_URL + "/Gene/query.json?criteria=products[abbreviation$eq'HumanMA']&num_rows=all",
        "factors": BASE_URL + "/query.json?criteria=model::Donor,rma::criteria,products[id$eq2],rma::include,age,rma::options[only$eq%27donors.id,dono  rs.name,donors.race_only,donors.sex%27]",
    }

    # there is a 1:1 mapping between donors and specimen for the 6 adult human brains
    _DONOR_IDS = ["15496", "14380", "15697", "9861", "12876", "10021"]
    _SPECIMEN_IDS = [
        "H0351.1015",
        "H0351.1012",
        "H0351.1016",
        "H0351.2001",
        "H0351.1009",
        "H0351.2002",
    ]

    def __init__(self, **kwargs):
        """
        Retrieves probes IDs for the given gene, then collects the
        Microarray probes, samples and z-scores for each donor.
        TODO check that this is only called for ICBM space
        """
        FeatureQuery.__init__(self)

        self.gene = kwargs.get('gene', None)
        self._specimen = {}
        self.factors = {}

        if self.gene is None:
            logger.warning(
                f'No gene name provided to {self.__class__.__name__}, so no gene expressions will be retrieved. '
                'Use the "gene=<name>" option in the feature query to specify one.')

        else:
            if not self.__class__._notification_shown:
                print(self.__class__.ALLEN_ATLAS_NOTIFICATION)
                self.__class__._notification_shown = True
            logger.info("Retrieving probe ids for gene {}".format(self.gene))
            url = self._QUERY["probe"].format(gene=self.gene)
            response = HttpRequest(url).get()
            if "site unavailable" in response.decode().lower():
                # When the Allen site is not available, they still send a status code 200.
                raise RuntimeError(f"Allen institute site unavailable - please try again later.")
            root = ElementTree.fromstring(response)
            num_probes = int(root.attrib["total_rows"])
            probe_ids = [int(root[0][i][0].text) for i in range(num_probes)]

            # get specimen information
            self._specimen = {
                spcid: self._retrieve_specimen(spcid) for spcid in self._SPECIMEN_IDS
            }
            response = HttpRequest(self._QUERY["factors"]).get()
            self.factors = {
                item["id"]: {
                    "race": item["race_only"],
                    "gender": item["sex"],
                    "age": int(item["age"]["days"] / 365),
                }
                for item in response["msg"]
            }

            # get expression levels and z_scores for the gene
            if len(probe_ids)>0:
                for donor_id in self._DONOR_IDS:
                    self._retrieve_microarray(donor_id, probe_ids)

    def _retrieve_specimen(self, specimen_id):
        """
        Retrieves information about a human specimen.
        """
        url = self._QUERY["specimen"].format(specimen_id=specimen_id)
        response = HttpRequest(url).get()
        if not response["success"]:
            raise Exception(
                "Invalid response when retrieving specimen information: {}".format(url)
            )
        # we ask for 1 specimen, so list should have length 1
        assert len(response["msg"]) == 1
        specimen = response["msg"][0]
        T = specimen["alignment3d"]
        specimen["donor2icbm"] = np.array(
            [
                [T["tvr_00"], T["tvr_01"], T["tvr_02"], T["tvr_09"]],
                [T["tvr_03"], T["tvr_04"], T["tvr_05"], T["tvr_10"]],
                [T["tvr_06"], T["tvr_07"], T["tvr_08"], T["tvr_11"]],
            ]
        )
        return specimen

    def _retrieve_microarray(self, donor_id, probe_ids):
        """
        Retrieve microarray data for several probes of a given donor, and
        compute the MRI position of the corresponding tissue block in the ICBM
        152 space to generate a SpatialFeature object for each sample.
        """

        if len(probe_ids)==0:
            return

        # query the microarray data for this donor
        url = self._QUERY["microarray"].format(
            probe_ids=",".join([str(id) for id in probe_ids]), donor_id=donor_id
        )
        response = HttpRequest(url, json.loads).get()
        if not response["success"]:
            raise Exception(
                "Invalid response when retrieving microarray data: {}".format(url)
            )

        # store probes
        probes, samples = [response["msg"][n] for n in ["probes", "samples"]]

        # store samples. Convert their MRI coordinates of the samples to ICBM
        # MNI152 space
        for i, sample in enumerate(samples):

            # coordinate conversion to ICBM152 standard space
            donor = {k: sample["donor"][k] for k in ["name", "id"]}
            icbm_coord = np.matmul(
                self._specimen[donor["name"]]["donor2icbm"],
                sample["sample"]["mri"] + [1],
            ).T

            # Create the spatial feature
            self.register(
                GeneExpression(
                    self.gene,
                    Point(icbm_coord, Space.REGISTRY.MNI152_2009C_NONL_ASYM),
                    expression_levels=[float(p["expression_level"][i]) for p in probes],
                    z_scores=[float(p["z-score"][i]) for p in probes],
                    probe_ids=[p["id"] for p in probes],
                    donor_info={**self.factors[donor["id"]], **donor},
                    mri_coord=sample["sample"]["mri"],
                    structure=sample['structure'],
                    top_level_structure=sample['top_level_structure'],
                )
            )


if __name__ == "__main__":

    featureextractor = AllenBrainAtlasQuery("GABARAPL2")
