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
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import tabular
from .. import anchor as _anchor
from ...core import concept as _concept, region as _region, parcellation as _parcellation
from ...retrieval import requests


# --------------------------------------------------------------------------- #
# Plotting helpers shared by both features
# --------------------------------------------------------------------------- #
def _wrapped_labels(names: Iterable, width: int, sep: str) -> Tuple[List[str], List[int]]:
    """Labels wrapped to `width` characters and joined with `sep`, and the
    number of lines of each label."""
    wrapped = [wrap(str(name), width) or [""] for name in names]
    return [sep.join(lines) for lines in wrapped], [len(lines) for lines in wrapped]


def _matplotlib_top_colorbar(fig, ax, mappable, label: str):
    """A slim horizontal colorbar above `ax`."""
    return fig.colorbar(
        mappable, ax=ax, location="top", label=label, shrink=0.6, pad=0.01, aspect=40
    )


def _plotly_title_and_top_colorbar(title: str) -> dict:
    """
    Layout for a title in the top left corner and a slim horizontal colorbar
    above the plot. The colorbar has no title of its own, because one there
    can hide the figure title; units go on an axis or in the hover text.
    """
    return dict(
        title=dict(
            text=title.replace("\n", "<br>"),
            x=0.01, xanchor="left", xref="container",
            y=0.99, yanchor="top", yref="container",
        ),
        coloraxis_colorbar=dict(
            orientation="h", x=0.5, xanchor="center", y=1.0, yanchor="bottom",
            len=0.6, thickness=10, title=dict(text=""),
        ),
    )


def _unsupported_backend(feature, backend: str) -> NotImplementedError:
    return NotImplementedError(
        f"Plotting {feature.__class__.__name__} is not supported for backend "
        f"'{backend}'; use 'matplotlib' or 'plotly'."
    )


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
class RegionalFunctionalActivation(
    tabular.Tabular,
    category="functional",
    configuration_folder="features/tabular/regional_functional_activation",
):
    """
    Mean activation of task-fMRI activation maps within every region of one
    parcellation.

    `data` has one row per activation map, indexed by its name, and one column
    per region of the parcellation, named by the region. A row gives one value
    per region, e.g. to colorize the parcellation map; a column is the
    functional fingerprint of that region (see :class:`FunctionalFingerprint`,
    served by a live query).

    Features of this type match queries for their parcellation, but not for
    individual regions, which receive functional fingerprints instead.
    """

    unit = "z"

    def __init__(
        self,
        file: str,
        decoder: Callable,
        anchor: _anchor.AnatomicalAnchor,
        description: str = "",
        modality: str = "regional functional activation",
        datasets: Optional[list] = None,
        id: Optional[str] = None,
        prerelease: bool = False,
    ):
        tabular.Tabular.__init__(
            self,
            modality=modality,
            description=description,
            anchor=anchor,
            data=None,  # loaded from `file` on first access, see `data`
            datasets=datasets or [],
            id=id,
            prerelease=prerelease,
        )
        self._loader = requests.HttpRequest(file, func=decoder)

    @property
    def data(self) -> pd.DataFrame:
        """Activation maps (rows) by regions (columns, named by the region)."""
        if self._data_cached is None:
            self._data_cached = self._loader.get()
        return self._data_cached.copy()

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

    def matches(self, concept: _concept.AtlasConcept) -> bool:
        # Individual regions are served by functional fingerprints (live
        # query); this feature only answers parcellation and space queries.
        if isinstance(concept, _region.Region) and not isinstance(
            concept, _parcellation.Parcellation
        ):
            self.anchor._last_matched_concept = None
            return False
        return super().matches(concept)

    def plot(self, *args, backend="matplotlib", label_width: int = 25, **kwargs):
        """
        Heatmap of the activation of every activation map (rows) in every
        region (columns).

        Parameters
        ----------
        backend: str
            "matplotlib" or "plotly"
        label_width: int
            Maximum number of characters per line of the activation map
            labels; longer names are wrapped onto several lines.
        **kwargs
            forwarded to `matplotlib.pyplot.imshow` or `plotly.express.imshow`
        """
        df = self.data
        regions = list(df.columns)
        title = kwargs.pop("title", self.name)

        if backend == "matplotlib":
            import matplotlib.pyplot as plt

            labels, line_counts = _wrapped_labels(df.index, label_width, "\n")
            height = max(4, 0.12 * sum(line_counts))
            width = max(6, 0.12 * len(regions))
            fig, ax = plt.subplots(figsize=(min(width, 30), min(height, 40)))
            image = ax.imshow(
                df.values, aspect="auto", cmap=kwargs.pop("cmap", "viridis"), **kwargs
            )
            ax.set_yticks(range(len(labels)), labels, fontsize="xx-small")
            ax.set_xticks(range(len(regions)), regions, rotation=90, fontsize="xx-small")
            # grid along cell boundaries
            ax.set_xticks(np.arange(-0.5, len(regions)), minor=True)
            ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.3)
            ax.tick_params(which="minor", length=0)
            _matplotlib_top_colorbar(fig, ax, image, self.unit)
            fig.suptitle(title, fontsize="medium")
            fig.tight_layout()
            return ax

        if backend == "plotly":
            import plotly.express as px

            labels, line_counts = _wrapped_labels(df.index, label_width, "<br>")
            # Rows are placed by position, so repeated activation map names
            # stay separate rows; only the displayed labels are wrapped, and
            # hovering shows the full names.
            positions = list(range(len(labels)))
            fig = px.imshow(
                df.values,
                x=regions,
                y=positions,
                aspect="auto",
                color_continuous_scale=kwargs.pop("color_continuous_scale", "viridis"),
                labels={"x": "region", "y": "activation map", "color": self.unit},
                **kwargs,
            )
            fig.update_traces(
                customdata=np.repeat(df.index.values[:, None], len(regions), axis=1),
                hovertemplate="%{customdata}<br>%{x}: %{z:.3f} " + self.unit + "<extra></extra>",
            )
            fig.update_layout(
                **_plotly_title_and_top_colorbar(title),
                height=max(400, 12 * sum(line_counts)) + 120,
                margin=dict(t=120),
                yaxis=dict(tickvals=positions, ticktext=labels, tickfont=dict(size=9)),
            )
            return fig

        raise _unsupported_backend(self, backend)


class FunctionalFingerprint(tabular.Tabular, category="functional"):
    """
    Mean activation of each task-fMRI activation map within one leaf region.

    Functional fingerprints are not configured individually. They are created
    on the fly by a live query from the :class:`RegionalFunctionalActivation`
    of the region's own parcellation, so a region only receives fingerprints
    computed for its parcellation version.
    """

    unit = RegionalFunctionalActivation.unit

    def __init__(self, region: _region.Region, func_activation: RegionalFunctionalActivation):
        """
        Parameters
        ----------
        region: Region
            A leaf region whose name is one of the columns of `func_activation`.
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
            anchor=_anchor.AnatomicalAnchor(region=region, species=list(func_activation.anchor.species)),
            data=None,  # cut from `func_activation` on access, see `data`
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
        name = self._region.name
        return self._func_activation.data[[name]].rename(columns={name: "mean"})

    def plot(self, *args, backend="matplotlib", label_width: int = 25, **kwargs):
        """
        Horizontal bar plot of the region's activation for every activation
        map, in the order of `data`, with bars colored by their value. Bars
        are placed by position, so repeated activation map names are shown as
        separate bars rather than pooled.

        Parameters
        ----------
        backend: str
            "matplotlib" or "plotly"
        label_width: int
            Maximum number of characters per line of the activation map
            labels; longer names are wrapped onto several lines.
        **kwargs
            forwarded to `pandas.Series.plot.barh` (matplotlib) or
            `plotly.express.bar` (plotly)
        """
        values = self.data["mean"]
        if values.notna().any():
            vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        else:
            vmin, vmax = 0.0, 1.0
        cmap = kwargs.pop("cmap", "viridis")
        title = kwargs.pop("title", "\n".join(wrap(
            f"{self.modality} of {self._region.name}", kwargs.pop("textwrap", 60)
        )))
        axis_label = f"mean ({self.unit})"

        if backend == "matplotlib":
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize

            labels, line_counts = _wrapped_labels(values.index, label_width, "\n")
            norm = Normalize(vmin=vmin, vmax=vmax)
            colormap = plt.get_cmap(cmap)
            # a new figure unless an axes is given; pandas would otherwise draw
            # into the current axes
            ax = kwargs.pop("ax", None)
            if ax is None:
                _, ax = plt.subplots(
                    figsize=kwargs.pop("figsize", (7, max(4, 0.2 * sum(line_counts))))
                )
            # pandas places bars by position (y = 0..n-1), so repeated names
            # stay separate bars
            values.plot.barh(
                ax=ax,
                color=[colormap(norm(v)) if pd.notna(v) else "lightgrey" for v in values],
                edgecolor="none", **kwargs
            )
            ax.set_yticks(range(len(labels)), labels)
            _matplotlib_top_colorbar(
                ax.figure, ax, plt.cm.ScalarMappable(norm=norm, cmap=colormap), axis_label
            )
            ax.figure.suptitle(title, fontsize="medium")
            ax.grid(axis="x", color="lightgrey", linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_xlabel(axis_label)
            ax.set_ylabel("")
            ax.invert_yaxis()
            ax.tick_params(axis="y", labelsize="xx-small")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.figure.tight_layout()
            return ax

        if backend == "plotly":
            import plotly.express as px

            labels, line_counts = _wrapped_labels(values.index, label_width, "<br>")
            # Bars are placed by position, so repeated activation map names stay
            # separate bars instead of being pooled into one category. Each bar
            # gets as many line units as its wrapped label has lines, so
            # multi-line labels do not overlap.
            counts = np.array(line_counts)
            positions = list(np.cumsum(counts) - counts / 2)
            fig = px.bar(
                x=values.values,
                y=positions,
                orientation="h",
                color=values.values,
                color_continuous_scale=cmap,
                range_color=(vmin, vmax),
                hover_name=values.index,
                labels={"x": axis_label, "y": "", "color": self.unit},
                **kwargs,
            )
            fig.update_traces(width=0.8, hovertemplate="%{hovertext}<br>%{x:.3f}<extra></extra>")
            fig.update_layout(
                **_plotly_title_and_top_colorbar(title),
                height=13 * sum(line_counts) + 220,
                margin=dict(t=120, b=70),
                xaxis=dict(automargin=True, title=dict(standoff=12)),
                yaxis=dict(
                    range=[sum(line_counts), 0],
                    tickvals=positions,
                    ticktext=labels,
                    tickfont=dict(size=9),
                ),
            )
            return fig

        raise _unsupported_backend(self, backend)
