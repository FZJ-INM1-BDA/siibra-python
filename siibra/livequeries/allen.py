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
"""Query Allen Human Brain Atlas microarray data in specified volume."""

from .query import LiveQuery

from ..core import space as _space
from ..features import anchor as _anchor
from ..features.tabular.gene_expression import GeneExpressions
from ..commons import logger, Species, MapType
from ..locations import Point, PointSet
from ..core.region import Region
from ..retrieval import HttpRequest
from ..vocabularies import GENE_NAMES

from typing import Iterable, List
from xml.etree import ElementTree
import numpy as np
import json


BASE_URL = "http://api.brain-map.org/api/v2/data"


class AllenBrainAtlasQuery(LiveQuery, args=['gene'], FeatureType=GeneExpressions):
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

    _FEATURETYPE = GeneExpressions

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
    species = Species.decode('homo sapiens')

    def __init__(self, **kwargs):
        """
        Each instance of this live query retrieves the probe IDs
        containing measurements for any gene in the given set
        of candidate genes.
        Each probe has expression levels and z-scores for a set of
        N samples.
        Each sample is linked to a donor, brain structure, and
        ICBM coordinate.
        When querying with a brain structure, the ICBM coordinates
        will be tested agains the region mask in ICBM space
        to produce a table of outputs.
        """
        LiveQuery.__init__(self, **kwargs)
        gene = kwargs.get('gene')
        self.maptype = kwargs.get("maptype", MapType.LABELLED)
        if isinstance(self.maptype, str):
            self.maptype = MapType[self.maptype.upper()]
        self.threshold_statistical = kwargs.get("threshold_statistical", 0)

        def parse_gene(spec):
            if isinstance(spec, str):
                return [GENE_NAMES.get(spec)]
            elif isinstance(spec, dict):
                assert all(k in spec for k in ['symbol', 'description'])
                assert spec['symbol'] in GENE_NAMES
                return [spec]
            elif isinstance(spec, list):
                return [g for s in spec for g in parse_gene(s)]
            else:
                logger.error("Invalid specification of gene:", spec)
                return []

        self.genes = parse_gene(gene)

    def query(self, region: Region) -> List[GeneExpressions]:
        assert isinstance(region, Region)
        space = _space.Space.registry().get('mni152')
        mask = region.fetch_regional_map(space, maptype=self.maptype, threshold=self.threshold_statistical)
        anchor = _anchor.AnatomicalAnchor(
            species=self.species, region=region.name
        )
        ass = _anchor.AnatomicalAssignment(
            query_structure=region,
            assigned_structure=region,
            qualification=_anchor.AssignmentQualification.CONTAINED,
            explanation=(f"MNI coordinates of tissue samples were compared with mask of '{region.name}' in {space}.")
        )
        anchor._assignments[region] = [ass]
        anchor._last_matched_concept = region
        anchor._location_cached = PointSet(coordinates=[], space=space)

        measures = []
        contained = {}
        for measure in self:
            location = Point(measure['mni_xyz'], space=space)
            if location not in contained:  # cache redundant intersection tests
                contained[location] = location.intersects(mask)
            if contained[location]:
                measures.append(measure)
                anchor._location_cached.points.append(location)

        yield GeneExpressions(
            anchor=anchor,
            genes=[m['gene'] for m in measures],
            levels=[m['expression_level'] for m in measures],
            z_scores=[m['z_score'] for m in measures],
            additional_columns={
                "race": [m['race'] for m in measures],
                "gender": [m['gender'] for m in measures],
                "age": [m['age'] for m in measures],
                "mni_xyz": [tuple(m['mni_xyz']) for m in measures],
                "sample": [m['sample_index'] for m in measures],
                "probe_id": [m['probe_id'] for m in measures],
                "donor_name": [m['donor_name'] for m in measures],
            }
        )

    def __iter__(self):

        if self.genes is None:
            logger.warning(
                f"No gene name provided to {self.__class__.__name__}, so no gene expressions will be retrieved. "
                'Use the "gene=<name>" option in the feature query to specify one.'
            )
            return

        if not self.__class__._notification_shown:
            print(GeneExpressions.ALLEN_ATLAS_NOTIFICATION)
            self.__class__._notification_shown = True

        assert isinstance(self.genes, list)
        if len(self.genes) == 1:
            logger.info(f"Retrieving probe ids for gene {self.genes[0]['symbol']}")
        else:
            logger.info(f"Retrieving probe ids for genes {', '.join(g['symbol'] for g in self.genes)}")
        url = self._QUERY["multiple_gene_probe"].format(genes=','.join([f"'{g['symbol']}'" for g in self.genes]))
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
                for item in self._retrieve_microarray(donor_id, probe_ids):
                    yield item

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
    def _retrieve_microarray(cls, donor_id: str, probe_ids: str) -> Iterable[GeneExpressions]:
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

        probes, samples = [response["msg"][n] for n in ["probes", "samples"]]

        for i, sample in enumerate(samples):

            # coordinate conversion to ICBM152 standard space
            donor_id = sample["donor"]["id"]
            donor_name = sample["donor"]["name"]
            icbm_coord = (np.matmul(
                AllenBrainAtlasQuery._specimen[donor_name]["donor2icbm"],
                sample["sample"]["mri"] + [1],
            )).round(2)

            for probe in probes:
                yield {
                    "gene": probe['gene-symbol'],
                    "expression_level": float(probe["expression_level"][i]),
                    "z_score": float(probe["z-score"][i]),
                    "sample_index": i,
                    "probe_id": probe["id"],
                    "donor_id": donor_id,
                    "donor_name": donor_name,
                    "race": cls.factors[donor_id]["race"],
                    "gender": cls.factors[donor_id]["gender"],
                    "age": cls.factors[donor_id]["age"],
                    "mni_xyz": icbm_coord[:3],
                }
