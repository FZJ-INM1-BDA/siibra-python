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
"""Streamlines in a reference space."""

from __future__ import annotations

from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from nibabel.streamlines import Tractogram as _NibTractogram

from .geometry import FragmentSpec, Geometry
from ..commons import logger
from ..core.region import Region

if TYPE_CHECKING:
    from trx.trx_file_memmap import TrxFile


class Tractogram(Geometry):
    """
    Streamlines in a reference space, such as the result of tractography.

    The fragments of a tractogram are its bundles, the groups of a TRX file.
    Streamlines are served memory-mapped through TRX: other formats, such
    as TRK, are converted to TRX once and cached.
    """

    def fetch(
        self,
        fragment: FragmentSpec = None,
        region: Union[str, Region] = None,
        format: str = None,
    ) -> Union[_NibTractogram, "TrxFile"]:
        """
        Load streamlines.

        Parameters
        ----------
        fragment: str or list of str, default: None
            Bundle(s) to load. A name matches exactly, or else as a unique
            case-insensitive substring of a bundle name.
        region: str or Region, default: None
            Load the bundles linked to this region or to its subregions.
            Cannot be combined with `fragment`.
        format: str, default: None
            - None or "tractogram": a nibabel Tractogram in the space, in mm.
              Its streamlines share the memory map of the cached file, unless
              an affine was applied.
            - "trx": the TRX file as stored, without applying an affine. Without
              a selection, this is siibra's loaded TRX file itself; do not
              close or modify it. A selection carries no groups.

        Selected bundles may share streamlines; each is returned once.
        """
        return super().fetch(fragment=fragment, region=region, format=format)

    def _format(self, trx: "TrxFile", format: Optional[str]):
        if format in (None, "tractogram"):
            tractogram = _NibTractogram(trx.streamlines, affine_to_rasmm=np.eye(4))
            if self._affine is not None:
                # copy first: never write into the memory-mapped cache file
                tractogram = tractogram.copy().apply_affine(self._affine)
            return tractogram
        if format == "trx":
            if self._affine is not None:
                logger.warning(f"Returning the TRX file of {self} as stored; its affine is not applied.")
            return trx
        raise ValueError(f"{self.__class__.__name__} supports the formats 'tractogram' and 'trx', not '{format}'.")
