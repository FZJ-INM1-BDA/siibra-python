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
"""Base type of features in tabular formats."""

from zipfile import ZipFile
from typing import Any, Callable, ClassVar, Dict, Optional, Set, TypedDict, Union
from dataclasses import dataclass, field
import json
import warnings

import pandas as pd
from textwrap import wrap

from .. import feature
from .. import anchor as _anchor
from ...commons import logger
from ...retrieval import requests


class UnitSpec(TypedDict, total=False):
    """Unit and description of one column or row of values in a tabular feature."""
    unit: str
    description: str
    long_name: str


@dataclass
class _Units:
    """
    Unit specs of a table, keyed by column or row label. The "*" entry is the
    default for numeric columns without an entry of their own. If the units are
    only known once the data is loaded, `loader` provides them on first use.
    """

    _BIDS_KEYS: ClassVar[Dict[str, str]] = {"unit": "Units", "description": "Description", "long_name": "LongName"}

    specs: Dict[str, UnitSpec] = field(default_factory=dict)
    loader: Optional[Callable[[], Dict[str, UnitSpec]]] = None

    @classmethod
    def from_bids_sidecar(cls, sidecar: dict) -> "_Units":
        """Column entries of a (merged) BIDS sidecar, with siibra's key names."""
        return cls({
            label: {key: entry[bids] for key, bids in cls._BIDS_KEYS.items() if bids in entry}
            for label, entry in (sidecar or {}).items()
            if isinstance(entry, dict)
        })

    def resolved(self) -> Dict[str, UnitSpec]:
        """The specs, loaded first if they are read from the data file."""
        if self.loader is not None:
            self.specs, self.loader = self.loader(), None
        return self.specs

    def spec_of(self, label) -> UnitSpec:
        specs = self.resolved()
        return {**specs.get("*", {}), **specs.get(label, {})}

    def distinct_units(self) -> Set[str]:
        return {spec["unit"] for spec in self.resolved().values() if "unit" in spec}

    def expand(self, data: pd.DataFrame) -> Dict[Any, UnitSpec]:
        """
        Specs of the labels in `data` that have one. Numeric columns get the
        default; any column or row label with an entry of its own is added.
        Columns of names or other labels get no default.
        """
        expanded = {column: self.spec_of(column) for column in data.select_dtypes("number").columns}
        for label in self.resolved():
            if label != "*" and (label in data.columns or label in data.index):
                expanded[label] = self.spec_of(label)
        return {label: spec for label, spec in expanded.items() if spec}

    def _to_bids(self, spec: UnitSpec) -> dict:
        return {bids: spec[key] for key, bids in self._BIDS_KEYS.items() if key in spec}

    def to_bids_sidecar(self, data: pd.DataFrame) -> dict:
        """
        Column entries of `data` with BIDS key names. BIDS sidecars describe
        columns only and have no default entry, so "*" is expanded per column
        and units of rows are left out.
        """
        return {
            str(label): self._to_bids(spec)
            for label, spec in self.expand(data).items()
            if label in data.columns
        }

    def to_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """One row per column or row label of `data`, with BIDS key names as columns."""
        frame = pd.DataFrame.from_dict(
            {label: self._to_bids(spec) for label, spec in self.expand(data).items()},
            orient="index",
        )
        return frame.reindex(columns=list(self._BIDS_KEYS.values()))


class Tabular(feature.Feature, category="generic", configuration_folder="features/tabular"):
    """
    Represents a table of different measures anchored to a brain location.

    Columns represent different types of values, while rows represent different
    samples. The number of columns might thus be interpreted as the feature
    dimension.

    As an example, receptor fingerprints use rows to represent different
    neurotransmitter receptors, and separate columns for the mean and standard
    deviations measure across multiple tissue samples.
    """

    def __init__(
        self,
        description: str,
        modality: str,
        anchor: _anchor.AnatomicalAnchor,
        file: str = None,
        decoder: Callable = None,
        data: pd.DataFrame = None,  # sample x feature dimension
        datasets: list = [],
        id: str = None,
        prerelease: bool = False,
        unit: Union[Dict[str, UnitSpec], Callable[[], Dict[str, UnitSpec]]] = None,
    ):
        """
        Parameters
        ----------
        unit: dict or callable, default: None
            Unit specs keyed by column or row label, with "*" as the default
            for all other values, e.g. ``{"*": {"unit": "mm"}}``. A callable
            returning such a dict is evaluated on first use, for units that
            are read from the data file.
        """
        feature.Feature.__init__(
            self,
            modality=modality,
            description=description,
            anchor=anchor,
            datasets=datasets,
            id=id,
            prerelease=prerelease
        )
        self._loader = None if file is None else requests.HttpRequest(file, func=decoder)
        if file is not None:
            assert data is None
        self._data_cached = data
        self._units = (
            _Units(loader=unit) if callable(unit)
            else _Units({label: dict(spec) for label, spec in (unit or {}).items()})
        )

    @property
    def unit(self) -> Optional[str]:
        """
        Deprecated: use `unit_of(label)` or `units` instead.

        The unit of the values if they all share one, else a ValueError.
        """
        warnings.warn(
            f"{type(self).__name__}.unit is deprecated and will be removed; use "
            "unit_of(label) for one column or row, or units for all of them.",
            FutureWarning,
            stacklevel=2,
        )
        units = self._units.distinct_units()
        if len(units) > 1:
            raise ValueError(
                f"{self.name} has several units ({', '.join(sorted(units))}); "
                "use unit_of(label) or units instead."
            )
        return next(iter(units), None)

    def unit_of(self, label) -> Optional[str]:
        """Unit of a column or row of values."""
        return self._units.spec_of(label).get("unit")

    @property
    def units(self) -> pd.DataFrame:
        """
        Unit, description and long name of each column or row of values, with
        the BIDS sidecar names (Units, Description, LongName) as columns.
        """
        return self._units.to_frame(self._load_data())

    @property
    def data(self) -> pd.DataFrame:
        """
        The table of values. `data.attrs["units"]` maps each column or row label
        with a known unit to that unit. It describes the data as returned, so it
        no longer holds once the values are transformed.
        """
        data = self._load_data()
        data.attrs["units"] = {
            label: spec["unit"] for label, spec in self._units.expand(data).items() if "unit" in spec
        }
        return data

    def _load_data(self) -> pd.DataFrame:
        """Load or compute the table. Derived classes override this instead of `data`."""
        if self._loader is not None:
            self._data_cached = self._loader.get()
        return self._data_cached.copy()

    def _to_zip(self, fh: ZipFile):
        super()._to_zip(fh)
        data = self.data
        fh.writestr("tabular.csv", data.to_csv())
        sidecar = self._units.to_bids_sidecar(data)
        if sidecar:
            fh.writestr("tabular.json", json.dumps(sidecar, indent=2))

    def plot(self, *args, backend="matplotlib", **kwargs):
        """
        Create a bar plot of a columns of the data.
        Parameters
        ----------
        backend: str
            "matplotlib", "plotly", or others supported by pandas DataFrame
            plotting backend.
        **kwargs
            takes Matplotlib.pyplot keyword arguments
        """
        wrapwidth = kwargs.pop("textwrap") if "textwrap" in kwargs else 40
        kwargs["title"] = kwargs.get(
            "title",
            "\n".join(wrap(
                f"{self.modality} in {', '.join({_.name for _ in self.anchor.regions})}",
                wrapwidth
            ))
        )
        kwargs["kind"] = kwargs.get("kind", "bar")
        kwargs["y"] = kwargs.get("y", self.data.columns[0])
        unit = self.unit_of(kwargs["y"])
        if backend == "matplotlib":
            try:
                import matplotlib.pyplot as plt
            except ImportError as e:
                logger.error(
                    "matplotlib not available. Please install matplotlib or use or another backend such as plotly."
                )
                raise e
            # default kwargs
            if kwargs.get("error_y") is None:
                kwargs["yerr"] = kwargs.get("yerr", 'std' if 'std' in self.data.columns else None)
                yerr_label = f" \u00b1 {kwargs.get('yerr')}" if kwargs.get('yerr') else ''
            if kwargs.get('kind') == 'bar':
                kwargs["width"] = kwargs.get("width", 0.8)
                kwargs["edgecolor"] = kwargs.get('edgecolor', 'black')
                kwargs["linewidth"] = kwargs.get('linewidth', 1.0)
                kwargs["capsize"] = kwargs.get('capsize', 4)
            kwargs["ylabel"] = kwargs.get(
                "ylabel",
                f"{kwargs['y']}{yerr_label}" + (f"\n{unit}" if unit else "")
            )
            kwargs["grid"] = kwargs.get("grid", True)
            kwargs["legend"] = kwargs.get("legend", False)
            kwargs["color"] = kwargs.get('color', 'darkgrey')

            xticklabel_rotation = kwargs.get("rot", 60)
            ax = self.data.plot(*args, backend=backend, **kwargs)
            ax.set_title(ax.get_title(), fontsize="medium")
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=xticklabel_rotation,
                ha='center' if xticklabel_rotation % 90 == 0 else 'right'
            )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            return ax
        elif backend == "plotly":
            kwargs["title"] = kwargs["title"].replace("\n", "<br>")
            kwargs["error_y"] = kwargs.get("error_y", 'std' if 'std' in self.data.columns else None)
            error_y_label = f" &plusmn; {kwargs.get('error_y')}<br>" if kwargs.get('error_y') else ''
            kwargs["labels"] = {
                "index": kwargs.pop("xlabel", None) or kwargs.pop("index", ""),
                "value": kwargs.pop("ylabel", None) or kwargs.pop(
                    "value",
                    f"{kwargs.get('y')}{error_y_label} {unit or ''}"
                )
            }
            fig = self.data.plot(*args, backend=backend, **kwargs)
            fig.update_layout(
                yaxis_title=kwargs["labels"]['value'],
                title=dict(
                    automargin=True, yref="container", xref="container",
                    pad=dict(t=40), xanchor="left", yanchor="top"
                )
            )
            return fig
        else:
            return self.data.plot(*args, backend=backend, **kwargs)
