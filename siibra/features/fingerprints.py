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

from typing import Union
import pandas as pd
from textwrap import wrap
import numpy as np
from io import BytesIO


class RegionalFingerprint(feature.Feature):
    """Represents a fingerprint of multiple variants of averaged measures in a brain region."""

    def __init__(
        self,
        description: str,
        modality: str,
        anchor: anchor.AnatomicalAnchor,
        means: Union[list, np.ndarray] = None,
        labels: Union[list, np.ndarray] = None,
        stds: Union[list, np.ndarray] = None,
        unit: str = None,
        datasets: list = []
    ):
        feature.Feature.__init__(
            self,
            modality=modality,
            description=description,
            anchor=anchor,
            datasets=datasets
        )
        self._means_cached = means
        self._labels_cached = labels
        self._stds_cached = stds
        self._unit = unit

    @property
    def unit(self):
        """Optionally overridden in derived class to allow lazy loading."""
        return self._unit

    @property
    def _labels(self):
        """Optionally overridden in derived class to allow lazy loading."""
        return self._labels_cached

    @property
    def _means(self):
        """Optionally overridden in derived class to allow lazy loading."""
        return self._means_cached

    @property
    def _stds(self):
        """Optionally overridden in derived class to allow lazy loading."""
        return self._stds_cached

    @property
    def data(self):
        return pd.DataFrame(
            {
                "mean": self._means,
                "std": self._stds,
            },
            index=self._labels,
        )

    def barplot(self, **kwargs):
        """Create a bar plot of the fingerprint."""

        wrapwidth = kwargs.pop("textwrap") if "textwrap" in kwargs else 40

        # default kwargs
        kwargs["width"] = kwargs.get("width", 0.95)
        kwargs["ylabel"] = kwargs.get("ylabel", self.unit)
        kwargs["title"] = kwargs.get("title", "\n".join(wrap(self.name, wrapwidth)))
        kwargs["grid"] = kwargs.get("grid", True)
        kwargs["legend"] = kwargs.get("legend", False)
        ax = self.data.plot(kind="bar", y="mean", yerr="std", **kwargs)
        ax.set_title(ax.get_title(), fontsize="medium")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")

    def plot(self, ax=None):
        """Create a polar plot of the fingerprint."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            commons.logger.error("matplotlib not available. Plotting of fingerprints disabled.")
            return None
        from collections import deque

        if ax is None:
            ax = plt.subplot(111, projection="polar")
        angles = deque(np.linspace(0, 2 * np.pi, len(self._labels) + 1)[:-1][::-1])
        angles.rotate(5)
        angles = list(angles)
        # for the values, repeat the first element to have a closed plot
        indices = list(range(len(self._means))) + [0]
        means = self.data["mean"].iloc[indices]
        stds0 = means - self.data["std"].iloc[indices]
        stds1 = means + self.data["std"].iloc[indices]
        plt.plot(angles + [angles[0]], means, "k-", lw=3)
        plt.plot(angles + [angles[0]], stds0, "k", lw=0.5)
        plt.plot(angles + [angles[0]], stds1, "k", lw=0.5)
        ax.set_xticks(angles)
        ax.set_xticklabels([_ for _ in self._labels])
        ax.set_title("\n".join(wrap(self.name, 40)))
        ax.tick_params(pad=9, labelsize=10)
        ax.tick_params(axis="y", labelsize=8)
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
            unit="detected cells / 0.1mm3",
        )
        self._filepairs = list(zip(segmentfiles, layerfiles))
        self._densities = None

    @property
    def densities(self):
        if self._densities is None:
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
            self._densities = pd.DataFrame(density_dict)
            self._densities.index.names = ["Layer"]
        return self._densities

    @property
    def _labels(self):
        return [profiles.CorticalProfile.LAYERS[_] for _ in self.densities.index]

    @property
    def _means(self):
        return self.densities.mean(axis=1).to_numpy()

    @property
    def _stds(self):
        return self.densities.std(axis=1).to_numpy()

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
        RegionalFingerprint.__init__(
            self,
            description=self.DESCRIPTION,
            modality="Modified silver staining",
            anchor=anchor,
            means=means,
            stds=stds,
            unit="staining intensity",
            labels=list(profiles.CorticalProfile.LAYERS.values())[1: -1],
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
            datasets=datasets,
        )

        self._data_cached = None
        self._loader = requests.HttpRequest(
            tsvfile,
            lambda url: self.parse_tsv_data(commons.decode_receptor_tsv(url)),
        )

    @property
    def unit(self):
        return self._loader.data['unit']

    @property
    def receptors(self):
        return self._loader.data['labels']

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
    def _labels(self):
        return self.receptors

    @property
    def _means(self):
        return self._loader.data['means']

    @property
    def _stds(self):
        return self._loader.data['stds']

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
