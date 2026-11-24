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
"""Streamline fiber bundles of a tractogram, stored as TRX."""

from contextlib import contextmanager
import logging
from typing import List, Optional, Union, TYPE_CHECKING
import weakref

import numpy as np

from .. import anchor as _anchor
from .. import feature
from ...commons import logger
from ...core import space as _space
from ...retrieval import requests

if TYPE_CHECKING:
    from trx.trx_file_memmap import TrxFile


class _GroupSelectionWarningFilter(logging.Filter):
    """trx-python warns on every selection that keeps groups, even without per-group data."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("Keeping dpg")


@contextmanager
def _without_group_selection_warning():
    root = logging.getLogger()  # trx-python logs through the root logger
    log_filter = _GroupSelectionWarningFilter()
    root.addFilter(log_filter)
    try:
        yield
    finally:
        root.removeFilter(log_filter)


class StreamlineFiberBundles(
    feature.Feature,
    configuration_folder="features/connectivity/streamlinefiberbundles",
    category="connectivity",
):
    """
    Streamline fiber bundles of a tractogram, stored in a TRX file whose
    groups are the bundles.

    Preconfigured features hold all bundles of a tractogram and are found by
    querying their space. Querying a region or parcellation returns features
    restricted to the bundles terminating in its leaf regions, computed by
    :class:`~siibra.livequeries.streamline_bundles.BundleFilterQuery`.
    """

    # Loaded TRX files by loader, so that filtered features share one memory map.
    _TRX_CACHE: "weakref.WeakKeyDictionary[requests.HttpRequest, TrxFile]" = weakref.WeakKeyDictionary()

    def __init__(
        self,
        modality: str,
        space_spec: dict,
        loader: requests.HttpRequest,
        anchor: _anchor.AnatomicalAnchor,
        name: str = "",
        description: str = "",
        bundles: List[str] = None,
        datasets: List = [],
        id: str = None,
        prerelease: bool = False,
    ):
        """
        Parameters
        ----------
        modality: str
        space_spec: dict
            Specification of the space of the streamlines, by "@id" or "name".
        loader: HttpRequest or FileLoader
            Loads the TRX file, decoding it with trx-python.
        anchor: AnatomicalAnchor
        name: str, default: ""
        description: str, default: ""
        bundles: list of str, default: None
            The bundles of this feature. None means all groups of the TRX file.
        datasets: list, default: []
        id: str, default: None
        prerelease: bool, default: False
        """
        feature.Feature.__init__(
            self,
            modality=modality,
            description=description,
            anchor=anchor,
            datasets=datasets,
            id=id,
            prerelease=prerelease,
        )
        self._space_spec = space_spec
        self._loader = loader
        self._name = name
        self._bundles: Optional[List[str]] = None if bundles is None else list(bundles)

    @property
    def name(self) -> str:
        if not self._name:
            return super().name
        name = f"{self._name} ({self.modality})"
        return f"[PRERELEASE] {name}" if self._prerelease else name

    @property
    def space(self) -> "_space.Space":
        for key in ["@id", "name"]:
            if key in self._space_spec:
                return _space.Space.get_instance(self._space_spec[key])
        return None

    def _load(self) -> "TrxFile":
        """The TRX file, loaded once per loader and memory-mapped."""
        if not requests._HAS_TRX:
            raise ImportError(
                "Reading streamline fiber bundles requires trx-python, which "
                "siibra installs on Python >= 3.11."
            )
        if self._loader not in self._TRX_CACHE:
            self._TRX_CACHE[self._loader] = self._loader.get()
        return self._TRX_CACHE[self._loader]

    @property
    def bundles(self) -> List[str]:
        """
        Names of the bundles of this feature.

        Features from a region query know their bundles without loading data.
        Otherwise the bundles are the groups of the TRX file, which is loaded
        to read them.
        """
        if self._bundles is None:
            self._bundles = sorted(self._load().groups.keys())
        return list(self._bundles)

    def fetch(self, bundles: Union[str, List[str]] = None) -> "TrxFile":
        """
        Load streamlines of bundles as a TRX file.

        Parameters
        ----------
        bundles: str or list of str, default: None
            Names of bundles of this feature, matched exactly. None loads all
            of them.

        Returns
        -------
        TrxFile
            The requested bundles as groups, each streamline once. If they are
            all bundles of the file, this is the file itself, as loaded by
            siibra; do not close it. A selection shares its memory map.
        """
        trx = self._load()
        available = self.bundles
        if bundles is None:
            names = available
        else:
            names = [bundles] if isinstance(bundles, str) else list(dict.fromkeys(bundles))
            unknown = [b for b in names if b not in available]
            if unknown:
                raise ValueError(f"{self.name} has no bundle(s) {unknown}. See its `bundles`.")

        missing = [b for b in names if b not in trx.groups]
        if missing:
            logger.warning(
                f"{len(missing)} bundle(s) of {self.name} are not groups of its "
                f"TRX file and are skipped: {', '.join(missing)}"
            )
            names = [b for b in names if b in trx.groups]
        if not names:
            raise ValueError(f"None of the requested bundles are in the TRX file of {self.name}.")
        if set(names) == set(trx.groups):
            return trx

        indices = np.unique(np.concatenate([np.asarray(trx.groups[b], dtype=np.int64) for b in names]))
        with _without_group_selection_warning():
            selection = trx.select(indices, keep_group=True)
        # bundles sharing streamlines with the requested ones would remain partially
        selection.groups = {b: g for b, g in selection.groups.items() if b in names}
        return selection

    def _filter_by_bundles(
        self, bundles: List[str], anchor: _anchor.AnatomicalAnchor
    ) -> "StreamlineFiberBundles":
        """A feature with only the given bundles of the same TRX file, and another anchor."""
        return StreamlineFiberBundles(
            modality=self.modality,
            space_spec=self._space_spec,
            loader=self._loader,
            anchor=anchor,
            name=self._name,
            description=self._description,
            bundles=bundles,
            datasets=self.datasets,
            id=self._id,
            prerelease=self._prerelease,
        )
