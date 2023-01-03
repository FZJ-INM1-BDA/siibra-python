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

from .. import _anchor
from .._basetypes import tabular

import pandas as pd
import numpy as np
from typing import List
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class GeneExpression(tabular.Tabular):
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

    def __init__(
        self,
        gene: str,
        expression_levels: List[float],
        z_scores: List[float],
        probe_ids: List[int],
        donor_info: DonorDict,
        anchor: _anchor.AnatomicalAnchor,
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
        data = pd.DataFrame(
            np.array([expression_levels, z_scores]).T,
            columns=['expression_level', 'z_score'],
            index=probe_ids
        )
        data.index.name = 'probe_id'
        tabular.Tabular.__init__(
            self,
            description=self.DESCRIPTION + self.ALLEN_ATLAS_NOTIFICATION,
            modality="Gene expression",
            anchor=anchor,
            data=data,
            datasets=datasets
        )
        self.donor_info = donor_info
        self.gene = gene
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
                + ",".join(["%4.1f" % v for v in self.data.expression_level])
                + "]",
                "Z-score: [" + ",".join(["%4.1f" % v for v in self.z_scores]) + "]",
            ]
        )
