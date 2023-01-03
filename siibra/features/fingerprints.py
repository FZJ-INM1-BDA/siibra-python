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

from . import feature, profiles, anchor

from .. import commons, vocabularies
from ..retrieval import requests

import pandas as pd
from textwrap import wrap
import numpy as np
from io import BytesIO
from typing import List
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class RegionalFingerprint(feature.Feature):
    """
    Represents a table of different measures anchored to a brain location.

    Columns represent different types of values, while rows represent
    different samples. The number of columns might thus be intrepreted
    as the feature dimension.

    As an example, receptor fingerprints use rows
    to represent different neurotransmitter receptors, and separate
    columns for the mean and standard deviations measure across multiple
    tissue samples.
    """

    def __init__(
        self,
        description: str,
        modality: str,
        anchor: anchor.AnatomicalAnchor,
        data: pd.DataFrame,  # sample x feature dimension
        datasets: list = []
    ):
        feature.Feature.__init__(
            self,
            modality=modality,
            description=description,
            anchor=anchor,
            datasets=datasets
        )
        self._data_cached = data

    @property
    def data(self):
        return self._data_cached

    def barplot(self, **kwargs):
        """Create a bar plot of the fingerprint."""

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            commons.logger.error("matplotlib not available. Plotting of fingerprints disabled.")
            return None

        wrapwidth = kwargs.pop("textwrap") if "textwrap" in kwargs else 40

        # default kwargs
        kwargs["y"] = kwargs.get("y", self.data.columns[0])
        kwargs["yerr"] = kwargs.get("yerr", 'std' if 'std' in self.data.columns else None)
        kwargs["width"] = kwargs.get("width", 0.95)
        kwargs["ylabel"] = kwargs.get("ylabel", self.data.columns[0])
        kwargs["title"] = kwargs.get(
            "title",
            "\n".join(wrap(f"{self.modality} anchored at {self.anchor._regionspec}", wrapwidth))
        )
        kwargs["grid"] = kwargs.get("grid", True)
        kwargs["legend"] = kwargs.get("legend", False)
        ax = self.data.plot(kind="bar", **kwargs)
        ax.set_title(ax.get_title(), fontsize="medium")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
        plt.tight_layout()

    def plot(self, y=None, yerr=None, ax=None):
        """ Create a polar plot of the fingerprint. """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            commons.logger.error("matplotlib not available. Plotting of fingerprints disabled.")
            return None
        from collections import deque

        if ax is None:
            ax = plt.subplot(111, projection="polar")

        datafield = y or self.data.columns[0]
        if yerr is None and 'std' in self.data.columns:
            yerr = 'std'
        # values = list(self.data[datafield])
        angles = deque(np.linspace(0, 2 * np.pi, self.data.shape[0] + 1)[:-1][::-1])
        angles.rotate(5)
        angles = list(angles)
        # for the values, repeat the first element to have a closed plot
        indices = list(range(self.data.shape[0])) + [0]
        y = list(self.data[datafield].iloc[indices])
        plt.plot(angles + [angles[0]], y, "k-", lw=3)
        if yerr:
            bounds0 = y - self.data[yerr].iloc[indices]
            plt.plot(angles + [angles[0]], bounds0, "k", lw=0.5)
            bounds1 = y + self.data[yerr].iloc[indices]
            plt.plot(angles + [angles[0]], bounds1, "k", lw=0.5)
        ax.set_xticks(angles)
        ax.set_xticklabels([_ for _ in self.data.index])
        ax.set_title(
            "\n".join(wrap(f"{self.modality} anchored at {self.anchor._regionspec}", 40))
        )
        ax.tick_params(pad=9, labelsize=10)
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        return ax


class CellDensityFingerprint(RegionalFingerprint, configuration_folder="features/fingerprints/celldensity"):

    DESCRIPTION = (
        "Layerwise estimated densities of detected cell bodies  (in detected cells per 0.1 cube millimeter) "
        "obtained by applying a Deep Learning based instance segmentation algorithm (Contour Proposal Network; Upschulte "
        "et al., Neuroimage 2022) to a 1 micron resolution cortical image patch prepared with modified Silver staining. "
        "Densities have been computed per cortical layer after manual layer segmentation, by dividing the number of "
        "detected cells in that layer with the area covered by the layer. Therefore, each profile contains 6 measurement points. "
        "The cortical depth is estimated from the measured layer thicknesses."
    )

    @classmethod
    def CELL_READER(cls, b):
        return pd.read_csv(BytesIO(b[2:]), delimiter=" ", header=0).astype(
            {"layer": int, "label": int}
        )

    @classmethod
    def LAYER_READER(cls, b):
        return pd.read_csv(BytesIO(b[2:]), delimiter=" ", header=0, index_col=0)

    def __init__(
        self,
        segmentfiles: list,
        layerfiles: list,
        anchor: anchor.AnatomicalAnchor,
        datasets: list = [],
    ):
        RegionalFingerprint.__init__(
            self,
            description=self.DESCRIPTION,
            modality="Segmented cell body density",
            anchor=anchor,
            datasets=datasets,
            data=None  # lazy loading below
        )
        self.unit = "detected cells / 0.1mm3",
        self._filepairs = list(zip(segmentfiles, layerfiles))
        self._densities = None

    def _load_densities(self):
        density_dict = {}
        for i, (cellfile, layerfile) in enumerate(self._filepairs):
            try:
                cells = requests.HttpRequest(cellfile, func=self.CELL_READER).data
                layers = requests.HttpRequest(layerfile, func=self.LAYER_READER).data
            except requests.SiibraHttpRequestError as e:
                print(str(e))
                commons.logger.error(f"Skipping to bootstrap a {self.__class__.__name__} feature, cannot access file resource.")
                continue
            counts = cells.layer.value_counts()
            areas = layers["Area(micron**2)"]
            density_dict[i] = counts[areas.index] / areas * 100 ** 2 * 5
        return pd.DataFrame(density_dict)
    
    @property
    def data(self):
        if self._data_cached is None:
            densities = self._load_densities()
            self._data_cached = pd.DataFrame(
                np.array([
                    list(densities.mean(axis=1)),
                    list(densities.std(axis=1))
                ]).T,
                columns=['mean', 'std'],
                index=[profiles.CorticalProfile.LAYERS[_] for _ in densities.index]
            )
            self._data_cached.index.name = 'layer'
        return self._data_cached

    @property
    def key(self):
        assert len(self.species) == 1
        return commons.create_key("{}_{}_{}".format(
            self.dataset_id,
            self.species[0]['name'],
            self.regionspec
        ))


class BigBrainIntensityFingerprint(RegionalFingerprint):

    DESCRIPTION = (
        "Layerwise averages and standard deviations of of BigBrain staining intensities "
        "computed by Konrad Wagstyl, as described in the publication "
        "'Wagstyl, K., et al (2020). BigBrain 3D atlas of "
        "cortical layers: Cortical and laminar thickness gradients diverge in sensory and "
        "motor cortices. PLoS Biology, 18(4), e3000678. "
        "http://dx.doi.org/10.1371/journal.pbio.3000678'."
        "Taken from the tutorial at https://github.com/kwagstyl/cortical_layers_tutorial "
        "and assigned to cytoarchitectonic regions of Julich-Brain."
    )

    def __init__(
        self,
        regionname: str,
        means: list,
        stds: list,
    ):

        from .anchor import AnatomicalAnchor
        anchor = AnatomicalAnchor(
            region=regionname,
            species='Homo sapiens'
        )
        data = pd.DataFrame(
            np.array([means, stds]).T,
            columns=['mean', 'std'],
            index=list(profiles.CorticalProfile.LAYERS.values())[1: -1]
        )
        data.index.name = "layer"
        RegionalFingerprint.__init__(
            self,
            description=self.DESCRIPTION,
            modality="Modified silver staining",
            anchor=anchor,
            data=data
        )


class ReceptorDensityFingerprint(RegionalFingerprint, configuration_folder="features/fingerprints/receptor"):

    DESCRIPTION = (
        "Fingerprint of densities (in fmol/mg protein) of receptors for classical neurotransmitters "
        "obtained by means of quantitative in vitro autoradiography. The fingerprint provides average "
        "density measurments for different receptors measured in tissue samples from different subjects "
        "together with the corresponding standard deviations. "
    )

    def __init__(
        self,
        tsvfile: str,
        anchor: anchor.AnatomicalAnchor,
        datasets: list = []
    ):
        """ Generate a receptor fingerprint from a URL to a .tsv file
        formatted according to the structure used by Palomero-Gallagher et al.
        """
        RegionalFingerprint.__init__(
            self,
            description=self.DESCRIPTION,
            modality="Neurotransmitter receptor density",
            anchor=anchor,
            data=None,  # lazy loading below
            datasets=datasets,
        )
        self._loader = requests.HttpRequest(
            tsvfile,
            lambda url: self.parse_tsv_data(commons.decode_receptor_tsv(url)),
        )

    @property
    def unit(self):
        return self._loader.data['unit']

    @property
    def receptors(self):
        return list(self.data.index)

    @property
    def neurotransmitters(self):
        return [
            "{} ({})".format(
                vocabularies.RECEPTOR_SYMBOLS[t]['neurotransmitter']['label'],
                vocabularies.RECEPTOR_SYMBOLS[t]['neurotransmitter']['name'],
            )
            for t in self.receptors
        ]
    
    @property
    def data(self):
        if self._data_cached is None:
            self._data_cached = pd.DataFrame(
                np.array([
                    self._loader.data['means'],
                    self._loader.data['stds']
                ]).T,
                index=self._loader.data['labels'],
                columns=['mean', 'std']
            )
            self._data_cached.index.name = 'receptor'
        return self._data_cached

    @property
    def key(self):
        return "{}_{}_{}_{}".format(
            commons.create_key(self.__class__.__name__),
            self.id,
            commons.create_key(self.species_name),
            commons.create_key(self.regionspec),
        )

    @classmethod
    def parse_tsv_data(cls, data: dict):
        units = {list(v.values())[3] for v in data.values()}
        labels = list(data.keys())
        assert len(units) == 1
        try:
            mean = [data[_]["density (mean)"] for _ in labels]
            std = [data[_]["density (sd)"] for _ in labels]
        except KeyError as e:
            print(str(e))
            commons.logger.error("Could not parse fingerprint from this dictionary")
        return {
            'unit': next(iter(units)),
            'labels': labels,
            'means': [float(m) if m.isnumeric() else 0 for m in mean],
            'stds': [float(s) if s.isnumeric() else 0 for s in std],
        }


class GeneExpression(RegionalFingerprint):
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
        anchor: anchor.AnatomicalAnchor,
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
            columns=['expression_level','z_score'],
            index=probe_ids
        )
        data.index.name = 'probe_id'
        RegionalFingerprint.__init__(
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
