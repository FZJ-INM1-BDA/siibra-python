# Copyright 2018-2024
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

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import List

from .point import Point
from .polyline import PolyLine
from .base import Location
from ...cache import fn_call_cache

X_PRECALCULATED_BOUNDARY_KEY = "x-siibra/layerboundary"
LAYERS = ("0", "I", "II", "III", "IV", "V", "VI", "WM")


@dataclass
class LayerBoundary(Location):
    schema: str = "siibra/attr/loc/layerboundary/v0.1"
    base_url: str = None

    @staticmethod
    @fn_call_cache
    def _GetPolyLineAttributes(url: str, space_id: str):
        import requests
        import numpy as np

        all_betweeners = (
            "0" if start == "0" else f"{start}_{end}"
            for start, end in zip(LAYERS[:-1], LAYERS[1:])
        )

        def return_segments(url):
            resp = requests.get(url)
            resp.raise_for_status()
            return resp.json().get("segments")

        def poly_srt(poly: np.ndarray):
            return poly[poly[:, 0].argsort(), :]

        with ThreadPoolExecutor() as ex:
            segments = ex.map(
                return_segments, (f"{url}{p}.json" for p in all_betweeners)
            )

        return [
            PolyLine(
                closed=False,
                points=[
                    Point(coord=coord, space_id=space_id)
                    for coord in poly_srt(np.array(s)).tolist()
                ],
                space_id=space_id,
            )
            for s in segments
        ]

    @property
    def layers(self) -> List[PolyLine]:
        if X_PRECALCULATED_BOUNDARY_KEY in self.extra:
            return self.extra[X_PRECALCULATED_BOUNDARY_KEY]
        return LayerBoundary._GetPolyLineAttributes(self.base_url, self.space_id)
