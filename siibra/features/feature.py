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

from ..commons import logger, InstanceTable
from ..core.concept import AtlasConcept

from typing import Union
import numpy as np
import pandas as pd
from textwrap import wrap
from tqdm import tqdm


class Feature:
    """
    Base class for anatomically anchored data features.
    """

    modalities = InstanceTable()

    def __init__(
        self,
        measuretype: str,
        description: str,
        anchor: "AnatomicalAnchor",
        datasets: list = []
    ):
        """
        Parameters
        ----------
        measuretype: str
            A textual description of the type of measured information
        description: str
            A textual description of the feature.
        anchor: AnatomicalAnchor
        datasets : list
            list of datasets corresponding to this feature
        """
        self.measuretype = measuretype
        self._description = description
        self.anchor = anchor
        self.datasets = datasets

    def __init_subclass__(cls, configuration_folder=None):
        cls.modalities.add(cls.__name__, cls)
        cls._live_queries = []
        cls._preconfigured_instances = None
        cls._configuration_folder = configuration_folder
        return super().__init_subclass__()

    @property
    def description(self):
        """ Allowssubclasses to overwrite the description with a function call. """
        return self._description

    @property
    def name(self):
        """Returns a short human-readable name of this feature."""
        return f"{self.__class__.__name__} ({self.measuretype}) anchored at {self.anchor}"

    @classmethod
    def get_instances(cls, **kwargs):
        """
        Retrieve objects of a particular feature subclass.
        Objects can be preconfigured in the configuration,
        or delivered by Live queries.
        """
        print(f"getting instances of {cls.__name__}")
        if cls._preconfigured_instances is None:
            if cls._configuration_folder is None:
                cls._preconfigured_instances = []
            else:
                from ..configuration import Configuration
                conf = Configuration()
                Configuration.register_cleanup(cls.clean_instances)
                assert cls._configuration_folder in conf.folders
                cls._preconfigured_instances = [
                    o for o in conf.build_objects(cls._configuration_folder)
                    if isinstance(o, cls)
                ]
                logger.debug(
                    f"Built {len(cls._preconfigured_instances)} preconfigured {cls.__name__} "
                    f"objects from {cls._configuration_folder}."
                )

        return cls._preconfigured_instances

    @classmethod
    def clean_instances(cls):
        """ Removes all instantiated object instances"""
        cls._preconfigured_instances = None

    def matches(self, concept: AtlasConcept) -> bool:
        if self.anchor and self.anchor.matches(concept):
            self._last_matched_concept = concept
            return True
        self._last_matched_concept = None
        return False

    @property
    def last_match_result(self):
        if self.anchor is None:
            return None
        return self.anchor._assignments.get(self._last_matched_concept)

    @classmethod
    def match(cls, concept: AtlasConcept, modality: Union[str, type], **kwargs):
        """
        Retrieve data features of the desired modality.
        """
        if isinstance(modality, str):
            modality = cls.modalities[modality]
        logger.info(f"Matching {modality.__name__} to {concept}")
        msg = f"Matching {modality.__name__} to {concept}"
        preconfigured_instances = [
            f for f in tqdm(modality.get_instances(), desc=msg)
            if f.matches(concept)
        ]

        live_instances = [] 
        for QueryType in modality._live_queries:
            logger.info(f"Running live query {QueryType.__name__} on {concept}")
            q = QueryType(**kwargs)
            live_instances.extend(q.query(concept, **kwargs))

        return preconfigured_instances + live_instances


# TODO how to allow rich text for label (e.g. markdown, latex) etc
class CorticalProfile(Feature):
    """
    Represents a 1-dimensional profile of measurements along cortical depth,
    measured at relative depths between 0 representing the pial surface,
    and 1 corresponding to the gray/white matter boundary.

    Mandatory attributes are the list of depth coordinates and the list of
    corresponding measurement values, which have to be of equal length,
    as well as a unit and description of the measurements.

    Optionally, the depth coordinates of layer boundaries can be specified.

    Most attributes are modelled as properties, so dervide classes are able
    to implement lazy loading instead of direct initialiation.

    """

    LAYERS = {0: "0", 1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "WM"}
    BOUNDARIES = list(zip(list(LAYERS.keys())[:-1], list(LAYERS.keys())[1:]))

    def __init__(
        self,
        description: str,
        measuretype: str,
        anchor: "AnatomicalAnchor",
        depths: Union[list, np.ndarray] = None,
        values: Union[list, np.ndarray] = None,
        unit: str = None,
        boundary_positions: dict = None,
        datasets: list = []
    ):
        """Initialize profile.

        Args:
            description (str):
                Human-readable of the modality of the measurements.
            measuretype (str):
                Short textual description of the modaility of measurements
            anchor: AnatomicalAnchor
            depths (list, optional):
                List of cortical depthh positions corresponding to each
                measurement, all in the range [0..1].
                Defaults to None.
            values (list, optional):
                List of the actual measurements at each depth position.
                Length must correspond to 'depths'.
                Defaults to None.
            unit (str, optional):
                Textual identifier for the unit of measurements.
                Defaults to None.
            boundary_positions (dict, optional):
                Dictionary of depths at which layer boundaries were identified.
                Keys are tuples of layer numbers, e.g. (1,2), values are cortical
                depth positions in the range [0..1].
                Defaults to None.
            datasets : list
                list of datasets corresponding to this feature
        """
        Feature.__init__(self, measuretype=measuretype, description=description, anchor=anchor, datasets=datasets)

        # cached properties will be revealed as property functions,
        # so derived classes may choose to override for lazy loading.
        self._unit = unit
        self._depths_cached = depths
        self._values_cached = values
        self._boundary_positions = boundary_positions

    def _check_sanity(self):
        # check plausibility of the profile
        assert isinstance(self._depths, (list, np.ndarray))
        assert isinstance(self._values, (list, np.ndarray))
        assert len(self._values) == len(self._depths)
        assert all(0 <= d <= 1 for d in self._depths)
        if self.boundaries_mapped:
            assert all(0 <= d <= 1 for d in self.boundary_positions.values())
            assert all(
                layerpair in self.BOUNDARIES
                for layerpair in self.boundary_positions.keys()
            )

    @property
    def unit(self):
        """Optionally overridden in derived classes."""
        if self._unit is None:
            raise NotImplementedError(f"'unit' not set for {self.__class__.__name__}.")
        return self._unit

    @property
    def boundary_positions(self):
        if self._boundary_positions is None:
            return {}
        else:
            return self._boundary_positions

    def assign_layer(self, depth: float):
        """Compute the cortical layer for a given depth from the
        layer boundary positions. If no positions are available
        for this profile, return None."""
        assert 0 <= depth <= 1
        if len(self.boundary_positions) == 0:
            return None
        else:
            return max(
                [l2 for (l1, l2), d in self.boundary_positions.items() if d < depth]
            )

    @property
    def boundaries_mapped(self) -> bool:
        if self.boundary_positions is None:
            return False
        else:
            return all((b in self.boundary_positions) for b in self.BOUNDARIES)

    @property
    def _layers(self):
        """List of layers assigned to each measurments,
        if layer boundaries are available for this features.
        """
        if self.boundaries_mapped:
            return [self.assign_layer(d) for d in self._depths]
        else:
            return None

    @property
    def data(self):
        """Return a pandas Series representing the profile."""
        self._check_sanity()
        return pd.Series(
            self._values, index=self._depths, name=f"{self.measuretype} ({self.unit})"
        )

    def plot(self, **kwargs):
        """Plot the profile.
        Keyword arguments are passed on to the plot command.
        'layercolor' can be used to specify a color for cortical layer shading.
        """
        wrapwidth = kwargs.pop("textwrap") if "textwrap" in kwargs else 40

        kwargs["title"] = kwargs.get("title", "\n".join(wrap(self.name, wrapwidth)))
        kwargs["xlabel"] = kwargs.get("xlabel", "Cortical depth")
        kwargs["ylabel"] = kwargs.get("ylabel", self.unit)
        kwargs["grid"] = kwargs.get("grid", True)
        kwargs["ylim"] = kwargs.get("ylim", (0, max(self._values)))
        layercolor = kwargs.pop("layercolor") if "layercolor" in kwargs else "black"
        axs = self.data.plot(**kwargs)

        if self.boundaries_mapped:
            bvals = list(self.boundary_positions.values())
            for i, (d1, d2) in enumerate(list(zip(bvals[:-1], bvals[1:]))):
                axs.text(
                    d1 + (d2 - d1) / 2.0,
                    10,
                    self.LAYERS[i + 1],
                    weight="normal",
                    ha="center",
                )
                if i % 2 == 0:
                    axs.axvspan(d1, d2, color=layercolor, alpha=0.1)

        axs.set_title(axs.get_title(), fontsize="medium")

        return axs

    @property
    def _depths(self):
        """Returns a list of the relative cortical depths of the measured values in the range [0..1].
        To be implemented in derived class."""
        if self._depths_cached is None:
            raise NotImplementedError(
                f"'_depths' not available for {self.__class__.__name__}."
            )
        return self._depths_cached

    @property
    def _values(self):
        """Returns a list of the measured values per depth.
        To be implemented in derived class."""
        if self._values_cached is None:
            raise NotImplementedError(
                f"'_values' not available for {self.__class__.__name__}."
            )
        return self._values_cached


class RegionalFingerprint(Feature):
    """Represents a fingerprint of multiple variants of averaged measures in a brain region."""

    def __init__(
        self,
        description: str,
        measuretype: str,
        anchor: "AnatomicalAnchor",
        means: Union[list, np.ndarray] = None,
        labels: Union[list, np.ndarray] = None,
        stds: Union[list, np.ndarray] = None,
        unit: str = None,
        datasets: list = []
    ):
        Feature.__init__(self, measuretype=measuretype, description=description, anchor=anchor, datasets=datasets)
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
            logger.error("matplotlib not available. Plotting of fingerprints disabled.")
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
