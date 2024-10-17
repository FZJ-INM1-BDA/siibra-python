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

from . import tabular

from .. import anchor as _anchor

import pandas as pd
from typing import Union, Dict, Tuple
from textwrap import wrap
import numpy as np


class CorticalProfile(tabular.Tabular):
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
        modality: str,
        anchor: _anchor.AnatomicalAnchor,
        depths: Union[list, np.ndarray] = None,
        values: Union[list, np.ndarray] = None,
        unit: str = None,
        boundary_positions: Dict[Tuple[int, int], float] = None,
        datasets: list = [],
        prerelease: bool = False,
        id: str = None,
    ):
        """Initialize profile.

        Parameters
        ----------
        description: str
            Human-readable of the modality of the measurements.
        modality: str
            Short textual description of the modality of measurements.
        anchor: AnatomicalAnchor
        depths: list, default: None
            List of cortical depth positions corresponding to each
            measurement, all in the range [0..1]
        values: list, default: None
            List of the actual measurements at each depth position.
            Length must correspond to 'depths'.
        unit: str, default: None
            Textual identifier for the unit of measurements.
        boundary_positions: dict, default: None
            Dictionary of depths at which layer boundaries were identified.
            Keys are tuples of layer numbers, e.g. (1,2), and values are
            cortical depth positions in the range [0..1].
        datasets : list[Dataset]
            list of datasets corresponding to this feature
        """

        # cached properties will be revealed as property functions,
        # so derived classes may choose to override for lazy loading.
        self._unit = unit
        self._depths_cached = depths
        self._values_cached = values
        self._boundary_positions = boundary_positions

        tabular.Tabular.__init__(
            self,
            modality=modality,
            description=description,
            anchor=anchor,
            data=None,  # lazy loader below
            datasets=datasets,
            prerelease=prerelease,
            id=id,
        )

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
    def unit(self) -> str:
        """Optionally overridden in derived classes."""
        if self._unit is None:
            raise NotImplementedError(f"'unit' not set for {self.__class__.__name__}.")
        return self._unit

    @property
    def boundary_positions(self) -> Dict[Tuple[int, int], float]:
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
        return pd.DataFrame(
            self._values, index=self._depths, columns=[f"{self.modality} ({self.unit})"]
        )

    def plot(self, *args, backend="matplotlib", **kwargs):
        """
        Plot the profile.

        Parameters
        ----------
        backend: str
            "matplotlib", "plotly", or others supported by pandas DataFrame
            plotting backend.
        **kwargs
            Keyword arguments are passed on to the plot command.
            'layercolor' can be used to specify a color for cortical layer shading.
        """
        wrapwidth = kwargs.pop("textwrap") if "textwrap" in kwargs else 40
        kwargs["title"] = kwargs.get("title", "\n".join(wrap(self.name, wrapwidth)))

        if backend == "matplotlib":    
            kwargs["xlabel"] = kwargs.get("xlabel", "Cortical depth")
            kwargs["ylabel"] = kwargs.get("ylabel", self.unit)
            kwargs["grid"] = kwargs.get("grid", True)
            kwargs["ylim"] = kwargs.get("ylim", (0, max(self._values)))
            layercolor = kwargs.pop("layercolor") if "layercolor" in kwargs else "black"
            axs = self.data.plot(*args, **kwargs, backend=backend)

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
        elif backend == "plotly":
            kwargs["title"] = kwargs["title"].replace("\n", "<br>")
            kwargs["labels"] = {
                "index": kwargs.pop("xlabel", None) or kwargs.pop("index", "Cortical depth"),
                "value": kwargs.pop("ylabel", None) or kwargs.pop("value", self.unit)
            }
            fig = self.data.plot(*args, **kwargs, backend=backend)
            if self.boundaries_mapped:
                bvals = list(self.boundary_positions.values())
                for i, (d1, d2) in enumerate(list(zip(bvals[:-1], bvals[1:]))):
                    fig.add_vrect(
                        x0=d1, x1=d2, line_width=0, fillcolor="gray",
                        opacity=0.2 if i % 2 == 0 else 0.0,
                        label=dict(text=self.LAYERS[i + 1], textposition="bottom center")
                    )
            fig.update_layout(
                showlegend=False,
                yaxis_range=(0, max(self._values)),
                title=dict(
                    automargin=True, yref="container", xref="container",
                    pad=dict(t=40), xanchor="left", yanchor="top"
                )
            )
            return fig
        else:
            return self.data.plot(*args, **kwargs, backend=backend)

    @property
    def _depths(self):
        """
        Returns a list of the relative cortical depths of the measured values in the range [0..1].

        To be implemented in derived class.
        """
        if self._depths_cached is None:
            raise NotImplementedError(
                f"'_depths' not available for {self.__class__.__name__}."
            )
        return self._depths_cached

    @property
    def _values(self):
        """
        Returns a list of the measured values per depth.

        To be implemented in derived class.
        """
        if self._values_cached is None:
            raise NotImplementedError(
                f"'_values' not available for {self.__class__.__name__}."
            )
        return self._values_cached
