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

from .. import anchor as _anchor
from ..basetypes import tabular

from ... import commons

import pandas as pd
from textwrap import wrap
from typing import List
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class GeneExpressions(tabular.Tabular):
    """
    A set gene expressions for different candidate genes
    measured inside a brain structure.
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

    class _DonorDict(TypedDict):
        id: int
        name: str
        race: str
        age: int
        gender: str

    class _SampleStructure(TypedDict):
        id: int
        name: str
        abbreviation: str
        color: str

    def __init__(
        self,
        levels: List[float],
        z_scores: List[float],
        genes: List[str],
        additional_columns: dict,
        anchor: _anchor.AnatomicalAnchor,
        datasets: List = []
    ):
        """
        Construct gene expression table.

        Parameters
        ----------
        levels : list of float
            Expression levels measured
        z_scores : list of float
            corresponding z scores measured
        genes : list of str
            Name of the gene corresponding to each measurement
        additional_columns : dict of list
            columns with additional data to be added to the tabular feature.
            Keys give column names, values are lists with the column data.
            Each list given needs to have the same length as expression_levels
        anchor: AnatomicalAnchor
        datasets : list
            list of datasets corresponding to this feature
        """
        assert len(z_scores) == len(levels)
        assert len(genes) == len(levels)
        if additional_columns is not None:
            assert all(len(lst) == len(levels) for lst in additional_columns.values())
        else:
            additional_columns = {}

        data = pd.DataFrame(
            dict(
                **{'level': levels, 'zscore': z_scores, 'gene': genes},
                **additional_columns
            )
        )
        # data.index.name = 'probe_id'
        tabular.Tabular.__init__(
            self,
            description=(
                (self.DESCRIPTION + self.ALLEN_ATLAS_NOTIFICATION)
                .replace('\n', ' ')
                .replace('\t', '')
                .strip()
            ),
            modality="Gene expression",
            anchor=anchor,
            data=data,
            datasets=datasets
        )
        self.unit = "expression level"

    def plot(self, **kwargs):
        """ Create a bar plot of the average per gene."""

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            commons.logger.error("matplotlib not available. Plotting of fingerprints disabled.")
            return None

        wrapwidth = kwargs.pop("textwrap") if "textwrap" in kwargs else 40

        for arg in ['yerr', 'y', 'ylabel', 'xlabel', 'width']:
            assert arg not in kwargs

        title = kwargs.pop("title", None) \
            or "\n".join(wrap(f"{self.modality} measured in {self.anchor._regionspec}", wrapwidth))
        kwargs["grid"] = kwargs.get("grid", True)
        kwargs["legend"] = kwargs.get("legend", False)

        # ax = plot_data.plot(kind="bar", **kwargs)
        ax = self.data.boxplot(column=['level'], by='gene', ax=kwargs.get('ax', None), showfliers=False)
        plt.title('')
        plt.suptitle('')
        ax.set_title(title, fontsize="medium")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
        ax.set_xlabel("")

        plt.tight_layout()
