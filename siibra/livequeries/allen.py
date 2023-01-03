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

from .query import LiveQuery

from ..core import space as _space
from ..features import anchor
from ..features.fingerprints import GeneExpression
from ..commons import logger, Species
from ..locations import Point
from ..core.region import Region
from ..retrieval import HttpRequest
from ..vocabularies import GENE_NAMES

from typing import Iterable, Union, List
from xml.etree import ElementTree
import numpy as np
import json


BASE_URL = "http://api.brain-map.org/api/v2/data"


class AllenBrainAtlasQuery(LiveQuery, args=['gene'], FeatureType=GeneExpression):
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

    _notification_shown = False

    _QUERY = {
        "probe": BASE_URL
        + "/query.xml?criteria=model::Probe,rma::criteria,[probe_type$eq'DNA'],products[abbreviation$eq'HumanMA'],gene[acronym$eq'{gene}'],rma::options[only$eq'probes.id']",
        "multiple_gene_probe": BASE_URL
        + "/query.xml?criteria=model::Probe,rma::criteria,[probe_type$eq'DNA'],products[abbreviation$eq'HumanMA'],gene[acronym$in{genes}],rma::options[only$eq'probes.id']",
        "specimen": BASE_URL
        + "/Specimen/query.json?criteria=[name$eq'{specimen_id}']&include=alignment3d",
        "microarray": BASE_URL
        + "/query.json?criteria=service::human_microarray_expression[probes$in{probe_ids}][donors$eq{donor_id}]",
        "gene": BASE_URL
        + "/Gene/query.json?criteria=products[abbreviation$eq'HumanMA']&num_rows=all",
        "factors": BASE_URL
        + "/query.json?criteria=model::Donor,rma::criteria,products[id$eq2],rma::include,age,rma::options[only$eq%27donors.id,dono  rs.name,donors.race_only,donors.sex%27]",
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

    _specimen = None
    factors = None
    space = _space.Space.registry().get('mni152')

    def __init__(self, **kwargs):
        """
        Retrieves probes IDs for the given gene, then collects the
        Microarray probes, samples and z-scores for each donor.
        TODO check that this is only called for ICBM space
        """
        LiveQuery.__init__(self, **kwargs)
        gene = kwargs.get('gene')

        def parse_gene(spec):
            if isinstance(spec, str):
                return GENE_NAMES.get(gene)
            elif isinstance(spec, dict):
                assert all(k in spec for k in ['symbol', 'description'])
                assert spec['symbol'] in GENE_NAMES
                return gene
            elif isinstance(spec, list):
                return [parse_gene(spec) for spec in gene]
            else:
                raise ValueError("Enexpected specification of gene: ", spec)

        self.gene = parse_gene(gene)

    def query(self, region: Region) -> List[GeneExpression]:
        assert isinstance(region, Region)
        mask = region.build_mask(self.space, "labelled")
        for f in self:
            if f.anchor.location.intersects(mask):
                # we construct the assignment manually,
                # although f.matches(region) would do it
                # for us, because the latter would add
                # signfiicant computational overhead for
                # re-doing the spatial assignment we
                # did already.
                ass = anchor.AnatomicalAssignment(
                    f.anchor.location,
                    region,
                    anchor.AssignmentQualification.CONTAINED,
                    explanation=(
                        f"{f.anchor.location} was compared with the mask "
                        f"of query region '{region.name}' in {self.space}."
                    )
                )
                f.anchor._assignments[region] = [ass]
                f.anchor._last_matched_concept = region
                yield f

    def __iter__(self):

        if self.gene is None:
            logger.warning(
                f"No gene name provided to {self.__class__.__name__}, so no gene expressions will be retrieved. "
                'Use the "gene=<name>" option in the feature query to specify one.'
            )
            return

        if not self.__class__._notification_shown:
            print(GeneExpression.ALLEN_ATLAS_NOTIFICATION)
            self.__class__._notification_shown = True

        logger.info("Retrieving probe ids for gene {}".format(self.gene['symbol']))
        url = self._QUERY["probe"].format(gene=self.gene['symbol'])
        if isinstance(self.gene, list):
            url = self._QUERY["multiple_gene_probe"].format(genes=','.join([f"'{g['symbol']}'" for g in self.gene]))
        response = HttpRequest(url).get()
        if "site unavailable" in response.decode().lower():
            # When the Allen site is not available, they still send a status code 200.
            raise RuntimeError(
                "Allen institute site unavailable - please try again later."
            )
        root = ElementTree.fromstring(response)
        num_probes = int(root.attrib["total_rows"])
        probe_ids = [int(root[0][i][0].text) for i in range(num_probes)]

        # get specimen information
        if AllenBrainAtlasQuery._specimen is None:
            AllenBrainAtlasQuery._specimen = {
                spcid: AllenBrainAtlasQuery._retrieve_specimen(spcid) for spcid in self._SPECIMEN_IDS
            }

        if AllenBrainAtlasQuery.factors is None:
            response = HttpRequest(self._QUERY["factors"]).get()
            AllenBrainAtlasQuery.factors = {
                item["id"]: {
                    "race": item["race_only"],
                    "gender": item["sex"],
                    "age": int(item["age"]["days"] / 365),
                }
                for item in response["msg"]
            }

        # get expression levels and z_scores for the gene
        if len(probe_ids) > 0:
            for donor_id in self._DONOR_IDS:
                if isinstance(self.gene, dict):
                    for gene_feature in AllenBrainAtlasQuery._retrieve_microarray(self.gene['symbol'], donor_id, probe_ids):
                        yield gene_feature
                else:
                    for gene in self.gene:
                        for gene_feature in AllenBrainAtlasQuery._retrieve_microarray(gene['symbol'], donor_id, probe_ids):
                            yield gene_feature

    @staticmethod
    def _retrieve_specimen(specimen_id: str):
        """
        Retrieves information about a human specimen.
        """
        url = AllenBrainAtlasQuery._QUERY["specimen"].format(specimen_id=specimen_id)
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

    @classmethod
    def _retrieve_microarray(cls, gene: str, donor_id: str, probe_ids: str) -> Iterable[GeneExpression]:
        """
        Retrieve microarray data for several probes of a given donor, and
        compute the MRI position of the corresponding tissue block in the ICBM
        152 space to generate a SpatialFeature object for each sample.
        """

        if len(probe_ids) == 0:
            return

        # query the microarray data for this donor
        url = AllenBrainAtlasQuery._QUERY["microarray"].format(
            probe_ids=",".join([str(id) for id in probe_ids]), donor_id=donor_id
        )
        response = HttpRequest(url, json.loads).get()
        if not response["success"]:
            raise Exception(
                "Invalid response when retrieving microarray data: {}".format(url)
            )

        # store probes
        probes, samples = [response["msg"][n] for n in ["probes", "samples"]]

        species = Species.decode('homo sapiens')

        # store samples. Convert their MRI coordinates of the samples to ICBM
        # MNI152 space
        for i, sample in enumerate(samples):

            # coordinate conversion to ICBM152 standard space
            donor = {k: sample["donor"][k] for k in ["name", "id"]}
            icbm_coord = np.matmul(
                AllenBrainAtlasQuery._specimen[donor["name"]]["donor2icbm"],
                sample["sample"]["mri"] + [1],
            ).T

            # Create the spatial feature
            yield GeneExpression(
                gene,
                expression_levels=[float(p["expression_level"][i]) for p in probes],
                z_scores=[float(p["z-score"][i]) for p in probes],
                probe_ids=[p["id"] for p in probes],
                donor_info={**AllenBrainAtlasQuery.factors[donor["id"]], **donor},
                anchor=anchor.AnatomicalAnchor(species=species, location=Point(icbm_coord, cls.space)),
                mri_coord=sample["sample"]["mri"],
                structure=sample["structure"],
                top_level_structure=sample["top_level_structure"],
            )
