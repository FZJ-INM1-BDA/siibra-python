# Copyright 2018-2026
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
"""Regional activation of task-fMRI activation maps, and the functional
fingerprints of single regions derived from it."""

from textwrap import wrap
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import tabular
from .. import anchor as _anchor
from ...commons import QUIET, logger
from ...core import region as _region, parcellation as _parcellation


class RegionalFunctionalActivation(
    tabular.Tabular,
    category="functional",
    configuration_folder="features/tabular/regional_functional_activation",
):
    """
    Mean activation of task-fMRI activation maps within every region of one
    parcellation.

    `data` has one row per activation map, indexed by its name, and one column
    per region of the parcellation. A row gives one value per region, e.g. to
    colorize the parcellation map; a column is the functional fingerprint of
    that region (see :class:`FunctionalFingerprint`, served by a live query).
    Descriptive columns of the activation maps listed in `METADATA_COLUMNS`
    are kept apart from the regional values in `activation_maps`.
    """

    unit = "z"
    METADATA_COLUMNS = ["task", "contrast", "control_condition", "target_condition"]

    def __init__(
        self,
        file: str,
        decoder: Callable,
        anchor: _anchor.AnatomicalAnchor,
        description: str = "",
        modality: str = "regional functional activation",
        datasets: list = [],
        id: str = None,
        prerelease: bool = False,
    ):
        tabular.Tabular.__init__(
            self,
            modality=modality,
            description=description,
            anchor=anchor,
            file=file,
            decoder=decoder,
            datasets=datasets,
            id=id,
            prerelease=prerelease,
        )
        self._values = None
        self._activation_maps = None

    @property
    def parcellation(self) -> _parcellation.Parcellation:
        """The parcellation whose regions are the columns of `data`."""
        parcellations = self.anchor.represented_parcellations()
        assert len(parcellations) == 1, (
            f"{self.__class__.__name__} must be anchored to exactly one "
            f"parcellation, found {parcellations}."
        )
        return parcellations[0]

    @property
    def name(self) -> str:
        return f"{super().name} ({self.parcellation.name})"

    def _load(self):
        df = self._loader.get()
        df.rename_axis(index={'pretty_name': "activation_map"})
        if not df.index.is_unique:
            duplicated = sorted(set(df.index[df.index.duplicated()]))
            logger.warning(
                f"{len(duplicated)} activation map name(s) of {self.name} are "
                f"not unique, so their rows cannot be addressed by name: {duplicated}"
            )
        metadata = [c for c in self.METADATA_COLUMNS if c in df.columns]
        self._activation_maps = df[metadata].copy() if metadata else None
        values = df.drop(columns=metadata)
        parc = self.parcellation
        regions = []
        with QUIET:
            for name in values.columns:
                try:
                    regions.append(parc.get_region(name))
                except Exception as e:
                    raise ValueError(
                        f"Column {name!r} of {self.name} is neither a metadata "
                        f"column ({self.METADATA_COLUMNS}) nor a region of {parc.name}."
                    ) from e
        values.columns = regions
        self._values = values.astype(float)

    @property
    def data(self) -> pd.DataFrame:
        """Activation maps (rows) by regions (columns, as region objects)."""
        if self._values is None:
            self._load()
        return self._values.copy()

    @property
    def activation_maps(self) -> Optional[pd.DataFrame]:
        """
        Descriptive columns of the activation maps (those of `METADATA_COLUMNS`
        present in the file), with the same index as `data`; None if the file
        has none of them.
        """
        if self._values is None:
            self._load()
        return None if self._activation_maps is None else self._activation_maps.copy()

    def plot(self, *args, backend="matplotlib", **kwargs):
        """
        Heatmap of the activation of every activation map (rows) in every
        region (columns).

        Parameters
        ----------
        backend: str
            "matplotlib" or "plotly"
        **kwargs
            forwarded to `matplotlib.pyplot.imshow` or `plotly.express.imshow`
        """
        df = self.data
        df.columns = [c.name for c in df.columns]
        title = kwargs.pop("title", self.name)
        if backend == "matplotlib":
            import matplotlib.pyplot as plt

            height = max(4, 0.12 * len(df.index))
            width = max(6, 0.12 * len(df.columns))
            fig, ax = plt.subplots(figsize=(min(width, 30), min(height, 40)))
            image = ax.imshow(
                df.values, aspect="auto",
                cmap=kwargs.pop("cmap", "viridis"), **kwargs
            )
            ax.set_yticks(range(len(df.index)), df.index, fontsize="xx-small")
            ax.set_xticks(
                range(len(df.columns)), df.columns,
                rotation=90, fontsize="xx-small"
            )
            ax.set_title(title, fontsize="medium")
            fig.colorbar(image, ax=ax, label=self.unit)
            fig.tight_layout()
            return ax
        elif backend == "plotly":
            import plotly.express as px

            fig = px.imshow(
                df,
                aspect="auto",
                color_continuous_scale=kwargs.pop("color_continuous_scale", "RdBu_r"),
                color_continuous_midpoint=kwargs.pop("color_continuous_midpoint", 0),
                labels={"x": "region", "y": "activation map", "color": self.unit},
                title=title,
                **kwargs,
            )
            fig.update_layout(height=max(400, 12 * len(df.index)))
            return fig
        else:
            raise NotImplementedError(
                f"Plotting {self.__class__.__name__} is not supported for backend "
                f"'{backend}'; use 'matplotlib' or 'plotly'."
            )


class FunctionalFingerprint(tabular.Tabular, category="functional"):
    """
    Mean activation of each task-fMRI activation map within one leaf region.

    Functional fingerprints are not configured individually. They are created
    on the fly by a live query from the :class:`RegionalFunctionalActivation`
    of the region's own parcellation, so a region only receives fingerprints
    computed for its parcellation version.
    """

    unit = "z"

    def __init__(self, region: _region.Region, func_activation: RegionalFunctionalActivation):
        """
        Parameters
        ----------
        region: Region
            A leaf region, one of the columns of `func_activation`.
        func_activation: RegionalFunctionalActivation
            The parcellation-wide feature this fingerprint is cut from.
        """
        tabular.Tabular.__init__(
            self,
            modality="functional fingerprint",
            description=(
                f"Functional fingerprint of {region.name}: mean activation of "
                f"each activation map within the region, from {func_activation.name}. "
                f"{func_activation.description}"
            ).strip(),
            anchor=_anchor.AnatomicalAnchor(region=region, species=func_activation.anchor.species),
            datasets=func_activation.datasets,
            id=f"{func_activation.id}--{region.key}",
            prerelease=func_activation._prerelease,
        )
        self._region = region
        self._func_activation = func_activation

    @property
    def region(self) -> _region.Region:
        return self._region

    @property
    def data(self) -> pd.DataFrame:
        """One row per activation map, with the region's mean activation."""
        return (
            self._func_activation.data[[self._region]]
            .rename(columns={self._region: "mean"})
        )

    @property
    def activation_maps(self) -> pd.DataFrame:
        """Descriptive columns of the activation maps; None if there are none."""
        return self._func_activation.activation_maps

    def plot(self, *args, backend="matplotlib", label_width: int = 30, **kwargs):
        """
        Horizontal bar plot of the region's activation for every activation
        map, with bars colored by their value on a diverging scale centered
        at zero.

        Parameters
        ----------
        backend: str
            "matplotlib" or "plotly"
        label_width: int
            Maximum number of characters per line of the activation map
            labels; longer names are wrapped onto several lines.
        **kwargs
            forwarded to `pandas.DataFrame.plot.barh` (matplotlib) or
            `plotly.express.bar` (plotly)
        """
        values = self.data["mean"]
        labels = [wrap(str(name), label_width) or [""] for name in values.index]
        n_lines = sum(len(lines) for lines in labels)
        limit = float(np.nanmax(np.abs(values))) if values.notna().any() else 1.0
        cmap = kwargs.pop("cmap", "viridis")
        title = kwargs.pop("title", "\n".join(wrap(
            f"{self.modality} of {self._region.name}", kwargs.pop("textwrap", 60)
        )))
        label = f"mean ({self.unit})"
        if backend == "matplotlib":
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize

            norm = Normalize(vmin=-limit, vmax=limit)
            colormap = plt.get_cmap(cmap)
            kwargs.setdefault("figsize", (7, max(4, 0.2 * n_lines)))
            ax = values.plot.barh(
                color=[colormap(norm(v)) for v in values.fillna(0)],
                edgecolor="none", title=title, **kwargs
            )
            # bars sit at y = 0..n-1; show wrapped names without changing the index
            ax.set_yticks(range(len(values)), ["\n".join(lines) for lines in labels])
            ax.figure.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax, label=label
            )
            ax.set_xlabel(label)
            ax.set_ylabel("")
            ax.invert_yaxis()
            ax.tick_params(axis="y", labelsize="xx-small")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.figure.tight_layout()
            return ax
        elif backend == "plotly":
            import plotly.express as px

            fig = px.bar(
                x=values.values,
                y=values.index,
                orientation="h",
                color=values.values,
                color_continuous_scale=cmap,
                range_color=(-limit, limit),
                labels={"x": label, "y": "", "color": self.unit},
                **kwargs,
            )
            fig.update_layout(
                title=dict(text=title.replace("\n", "<br>"), automargin=True, xanchor="left", yanchor="top"),
                height=max(400, 12 * n_lines),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(size=9),
                    # bars keep the full names as categories (and in hover);
                    # only the displayed tick labels are wrapped
                    tickvals=list(values.index),
                    ticktext=["<br>".join(lines) for lines in labels],
                ),
            )
            return fig
        else:
            raise NotImplementedError(
                f"Plotting {self.__class__.__name__} is not supported for backend "
                f"'{backend}'; use 'matplotlib' or 'plotly'."
            )
