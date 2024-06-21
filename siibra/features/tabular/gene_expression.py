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
from . import tabular
from ...retrieval.datasets import GenericDataset

import pandas as pd
from textwrap import wrap
from typing import List
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class GeneExpressions(
    tabular.Tabular,
    category='molecular'
):
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

    DATASET = GenericDataset(
        name="An anatomically comprehensive atlas of the adult human brain transcriptome",
        contributors=[
            'Michael J. Hawrylycz',
            'Ed S. Lein',
            'Angela L. Guillozet-Bongaarts',
            'Elaine H. Shen',
            'Lydia Ng',
            'Jeremy A. Miller',
            'Louie N. van de Lagemaat',
            'Kimberly A. Smith',
            'Amanda Ebbert',
            'Zackery L. Riley',
            'Chris Abajian',
            'Christian F. Beckmann',
            'Amy Bernard',
            'Darren Bertagnolli',
            'Andrew F. Boe',
            'Preston M. Cartagena',
            'M. Mallar Chakravarty',
            'Mike Chapin',
            'Jimmy Chong',
            'Rachel A. Dalley',
            'Barry David Daly',
            'Chinh Dang',
            'Suvro Datta',
            'Nick Dee',
            'Tim A. Dolbeare',
            'Vance Faber',
            'David Feng',
            'David R. Fowler',
            'Jeff Goldy',
            'Benjamin W. Gregor',
            'Zeb Haradon',
            'David R. Haynor',
            'John G. Hohmann',
            'Steve Horvath',
            'Robert E. Howard',
            'Andreas Jeromin',
            'Jayson M. Jochim',
            'Marty Kinnunen',
            'Christopher Lau',
            'Evan T. Lazarz',
            'Changkyu Lee',
            'Tracy A. Lemon',
            'Ling Li',
            'Yang Li',
            'John A. Morris',
            'Caroline C. Overly',
            'Patrick D. Parker',
            'Sheana E. Parry',
            'Melissa Reding',
            'Joshua J. Royall',
            'Jay Schulkin',
            'Pedro Adolfo Sequeira',
            'Clifford R. Slaughterbeck',
            'Simon C. Smith',
            'Andy J. Sodt',
            'Susan M. Sunkin',
            'Beryl E. Swanson',
            'Marquis P. Vawter',
            'Derric Williams',
            'Paul Wohnoutka',
            'H. Ronald Zielke',
            'Daniel H. Geschwind',
            'Patrick R. Hof',
            'Stephen M. Smith',
            'Christof Koch',
            'Seth G. N. Grant',
            'Allan R. Jones'
        ],
        url="https://doi.org/10.1038%2Fnature11405",
        description='Neuroanatomically precise, genome-wide maps of transcript '
            'distributions are critical resources to complement genomic '
            'sequence data and to correlate functional and genetic brain '
            'architecture. Here we describe the generation and analysis '
            'of a transcriptional atlas of the adult human brain, '
            'comprising extensive histological analysis and comprehensive '
            'microarray profiling of ~900 neuroanatomically precise '
            'subdivisions in two individuals. Transcriptional regulation '
            'varies enormously by anatomical location, with different '
            'regions and their constituent cell types displaying robust '
            'molecular signatures that are highly conserved between '
            'individuals. Analysis of differential gene expression and '
            'gene co-expression relationships demonstrates that brain-'
            'wide variation strongly reflects the distributions of major '
            'cell classes such as neurons, oligodendrocytes, astrocytes '
            'and microglia. Local neighbourhood relationships between '
            'fine anatomical subdivisions are associated with discrete '
            'neuronal subtypes and genes involved with synaptic '
            'transmission. The neocortex displays a relatively '
            'homogeneous transcriptional pattern, but with distinct '
            'features associated selectively with primary sensorimotor '
            'cortices and with enriched frontal lobe expression. Notably, '
            'the spatial topography of the neocortex is strongly '
            'reflected in its molecular topography— the closer two '
            'cortical regions, the more similar their transcriptomes. '
            'This freely accessible online data resource forms a high-'
            'resolution transcriptional baseline for neurogenetic studies '
            'of normal and abnormal human brain function.'
            ""
            "For retrieving microarray data, siibra connects to the web API of "
            "the Allen Brain Atlas (© 2015 Allen Institute for Brain Science), "
            "available from https://brain-map.org/api/index.html. Any use of the "
            "microarray data needs to be in accordance with their terms of use, "
            "as specified at https://alleninstitute.org/legal/terms-use/."
    )

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
        datasets: List = [DATASET]
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

        _data_cahced = pd.DataFrame(
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
            data=_data_cahced,
            datasets=datasets
        )
        self.unit = "expression level"

    def plot(self, *args, backend="matplotlib", **kwargs):
        """
        Create a box plot per gene.

        Parameters
        ----------
        backend: str
            "matplotlib", "plotly", or others supported by pandas DataFrame
            plotting backend.
        **kwargs
            Keyword arguments are passed on to the plot command.
        """
        wrapwidth = kwargs.pop("textwrap") if "textwrap" in kwargs else 40
        kwargs["title"] = kwargs.pop("title", None) \
            or "\n".join(wrap(f"{self.modality} measured in {self.anchor._regionspec or self.anchor.location}", wrapwidth))
        kwargs["kind"] = "box"
        if backend == "matplotlib":
            for arg in ['yerr', 'y', 'ylabel', 'xlabel', 'width']:
                assert arg not in kwargs
            default_kwargs = {
                "grid": True, "legend": False, 'by': "gene",
                'column': ['level'], 'showfliers': False, 'ax': None,
                'ylabel': 'expression level'
            }
            return self.data.plot(*args, **{**default_kwargs, **kwargs}, backend=backend)
        elif backend == "plotly":
            kwargs["title"] = kwargs["title"].replace('\n', "<br>")
            return self.data.plot(y='level', x='gene', backend=backend, **kwargs)
        else:
            return self.data.plot(*args, backend=backend, **kwargs)
