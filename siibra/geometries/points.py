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
"""Points in a reference space, with optional per-point attributes."""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from nibabel.affines import apply_affine

from .geometry import FragmentSpec, Geometry
from .providers.points import PointArrays
from ..core.region import Region


class Points(Geometry):
    """
    Points in a reference space, such as cell positions, with optional
    per-point attributes such as cell size.

    Fragments are typically the points of one region each, delivered in one
    file per fragment. Points are served memory-mapped from siibra's cache.
    """

    COORDINATE_COLUMNS = ["x", "y", "z"]

    def fetch(
        self,
        fragment: FragmentSpec = None,
        region: Union[str, Region] = None,
        format: str = None,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Load points.

        Parameters
        ----------
        fragment: str or list of str, default: None
            Fragment(s) to load. A name matches exactly, or else as a unique
            case-insensitive substring of a fragment name.
        region: str or Region, default: None
            Load the fragments linked to this region or to its subregions.
            Cannot be combined with `fragment`.
        format: str, default: None
            - None or "dataframe": a table with columns x, y and z in mm in
              the space, then the attributes. If the points have fragments,
              a categorical column "fragment" names each point's fragment.
            - "numpy": the coordinates as an (N, 3) array in mm in the space.
              It is memory-mapped, read-only, when it comes from a single
              file and no affine is applied.
        """
        return super().fetch(fragment=fragment, region=region, format=format)

    def _transform(self, coordinates: np.ndarray) -> np.ndarray:
        if self._affine is None:
            return coordinates
        return apply_affine(self._affine, np.asarray(coordinates))

    def _format(self, arrays: Dict[Optional[str], PointArrays], format: Optional[str]):
        if format == "numpy":
            coordinates = [c for c, _ in arrays.values()]
            return self._transform(coordinates[0] if len(coordinates) == 1 else np.concatenate(coordinates))
        if format in (None, "dataframe"):
            frames = []
            for name, (coordinates, attributes) in arrays.items():
                frame = pd.DataFrame(
                    np.array(self._transform(coordinates)), columns=self.COORDINATE_COLUMNS
                )
                if attributes is not None:
                    frame = frame.join(pd.DataFrame(np.asarray(attributes)))
                if name is not None:
                    frame["fragment"] = name
                frames.append(frame)
            table = pd.concat(frames, ignore_index=True)
            if "fragment" in table:
                table["fragment"] = pd.Categorical(table["fragment"], categories=list(arrays))
            return table
        raise ValueError(f"{self.__class__.__name__} supports the formats 'dataframe' and 'numpy', not '{format}'.")
