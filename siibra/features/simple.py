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

# simple data features anchored to a point in space

from .feature import Feature

from typing import List, TypedDict


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


class GeneExpression(Feature):
    """
    A spatial feature type for gene expressions.
    """

    DESCRIPTION = """
    Gene expressions extracted from microarray data in the Allen Atlas.
    """

    ALLEN_ATLAS_NOTIFICATION = """
    For retrieving microarray data, siibra connects to the web API of
    the Allen Brain Atlas (© 2015 Allen Institute for Brain Science),
    available from https://brain-map.org/api/index.html. Any use of the
    microarray data needs to be in accordance with their terms of use,
    as specified at https://alleninstitute.org/legal/terms-use/.
    """

    def __init__(
        self,
        gene: str,
        expression_levels: List[float],
        z_scores: List[float],
        probe_ids: List[int],
        donor_info: DonorDict,
        anchor: "AnatomicalAnchor",
        mri_coord: List[int] = None,
        structure: SampleStructure = None,
        top_level_structure: SampleStructure = None,
        datasets: List = []
    ):
        """
        Construct the spatial feature for gene expressions measured in a sample.

        Parameters
        ----------
        gene : str
            Name of gene
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
        anchor: AnatomicalAnchor
        datasets : list
            list of datasets corresponding to this feature
        """
        Feature.__init__(
            self,
            measuretype="Gene expression",
            description=self.DESCRIPTION + self.ALLEN_ATLAS_NOTIFICATION,
            anchor=anchor,
            datasets=datasets
        )
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
                "At (" + ",".join("{:4.0f}".format(v) for v in self.anchor.location) + ")",
                " ".join(
                    [
                        "{:>7.7}:{:7.7}".format(k, str(v))
                        for k, v in self.donor_info.items()
                    ]
                ),
                "Expression: ["
                + ",".join(["%4.1f" % v for v in self.expression_levels])
                + "]",
                "Z-score: [" + ",".join(["%4.1f" % v for v in self.z_scores]) + "]",
            ]
        )
