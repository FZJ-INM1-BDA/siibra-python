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
"""Providers of tractograms: TRX, and TRK converted to TRX."""

from __future__ import annotations

from abc import abstractmethod
import json
import os
from typing import Dict, List, Optional, Union, TYPE_CHECKING

import numpy as np
import nibabel as nib
from nibabel.streamlines import ArraySequence, Tractogram as _NibTractogram

from .provider import GeometryProvider, _convert_once, _local_file, _source_key

try:
    from trx import trx_file_memmap as _trx
except ImportError:  # trx-python is only installed on Python >= 3.11
    _trx = None

if TYPE_CHECKING:
    from trx.trx_file_memmap import TrxFile


def _require_trx():
    if _trx is None:
        raise ImportError(
            "Reading tractograms requires trx-python, which siibra installs "
            "on Python >= 3.11."
        )


class _TrxBackedProvider(GeometryProvider):
    """
    Serves a TRX file, memory-mapped. The TRX groups are the fragments.
    Subclasses define where the TRX file comes from.
    """

    def __init__(self):
        self._trx: Optional["TrxFile"] = None

    @abstractmethod
    def _trx_path(self) -> str:
        """Local path of the TRX file to serve."""
        raise NotImplementedError

    def _load(self) -> "TrxFile":
        if self._trx is None:
            _require_trx()
            self._trx = _trx.load(self._trx_path())
        return self._trx

    @property
    def fragments(self) -> List[str]:
        # sorted, since TRX does not define an order of its groups
        return sorted(self._load().groups.keys())

    def _indices(self, fragments: List[str]) -> np.ndarray:
        """Sorted indices of the streamlines in any of the given groups."""
        groups = self._load().groups
        unknown = [f for f in fragments if f not in groups]
        if unknown:
            raise ValueError(f"No group(s) {unknown} in tractogram {self._url}.")
        if not fragments:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate([np.asarray(groups[f], dtype=np.int64) for f in fragments]))

    def fetch(self, fragments: List[str] = None) -> "TrxFile":
        """
        Return the TRX file, or a selection of its streamlines.

        Without fragments, this is siibra's loaded TRX file itself; do not
        close or modify it. A selection shares its memory map and contains
        each streamline once, even if it is in several of the selected
        groups. It carries no groups.
        """
        trx = self._load()
        if fragments is None:
            return trx
        return trx.select(self._indices(fragments), keep_group=False)

    def get_coordinates(self, fragments: List[str] = None) -> np.ndarray:
        if fragments is None:
            return self._load().streamlines._data  # the memory-mapped positions
        return self.fetch(fragments).streamlines.get_data()


class TrxProvider(_TrxBackedProvider, srctype="trx"):
    """A tractogram in one TRX file, whose groups are the fragments."""

    def __init__(self, src: str):
        """
        Parameters
        ----------
        src: str
            URL or local path of a TRX file.
        """
        if not isinstance(src, str):
            raise ValueError(
                f"A TRX provider takes one URL or path, got {src!r}. "
                "Fragments are the groups inside the TRX file."
            )
        _TrxBackedProvider.__init__(self)
        self._src = src

    @property
    def _url(self) -> str:
        return self._src

    def _trx_path(self) -> str:
        return _local_file(self._src)


class TrkProvider(_TrxBackedProvider, srctype="trk"):
    """
    A tractogram in one or more TRK files, e.g. one per bundle.

    The files are converted once into a single TRX file in siibra's cache,
    where each file becomes a group named after its fragment, and then
    served memory-mapped like any TRX file.
    """

    def __init__(self, src: Union[str, Dict[str, str]]):
        """
        Parameters
        ----------
        src: str or dict
            URL or local path of a TRK file, or a mapping of fragment names
            to such sources.
        """
        if isinstance(src, str):
            self._sources: Dict[Optional[str], str] = {None: src}
        elif isinstance(src, dict) and len(src) > 0:
            self._sources = dict(src)
        else:
            raise ValueError(f"A TRK provider takes a URL or path, or a mapping of fragments to them, got {src!r}.")
        _TrxBackedProvider.__init__(self)
        self._src = src

    @property
    def _url(self) -> Union[str, Dict[str, str]]:
        return self._src

    def _trx_path(self) -> str:
        key = json.dumps(
            ["trk", [[name, _source_key(src)] for name, src in self._sources.items()]]
        )
        return _convert_once(key, ".trx", self._convert)

    def _convert(self, target: str):
        """Write the TRK files as one TRX file to `target`."""
        _require_trx()
        streamlines = ArraySequence()
        groups = {}
        reference = None
        for name, src in self._sources.items():
            trk = nib.streamlines.load(_local_file(src))  # streamlines in RAS+ mm
            if reference is None:
                reference = trk  # the first header defines the TRX header
            start = len(streamlines)
            streamlines.extend(trk.streamlines)
            if name is not None:
                groups[name] = np.arange(start, len(streamlines), dtype=np.uint32)
        trx = _trx.TrxFile.from_tractogram(
            _NibTractogram(streamlines, affine_to_rasmm=np.eye(4)), reference=reference
        )
        trx.groups = groups
        zipped = f"{target}.trx"  # trx-python chooses the zip format by extension
        try:
            _trx.save(trx, zipped)  # uncompressed, so that it can be memory-mapped
            os.replace(zipped, target)
        finally:
            trx.close()
            if os.path.exists(zipped):
                os.remove(zipped)
