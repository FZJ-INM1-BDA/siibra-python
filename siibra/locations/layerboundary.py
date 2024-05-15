from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import List

from .point import Pt
from .polyline import Polyline
from .base import Location
from ..cache import fn_call_cache

@dataclass
class LayerBoundary(Location):
    schema: str = "siibra/attr/loc/layerboundary"
    url: str = None

    @staticmethod
    @fn_call_cache
    def _GetPolylineAttributes(url: str, space_id: str):
        import requests
        import numpy as np

        LAYERS = ("0", "I", "II", "III", "IV", "V", "VI", "WM")

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
            Polyline(
                closed=False,
                points=[Pt(coord=coord, space_id=space_id)
                        for coord in poly_srt(np.array(s)).tolist()],
                space_id=space_id,
            )
            for s in segments
        ]

    @property
    def layers(self) -> List[Polyline]:
        return LayerBoundary._GetPolylineAttributes(self.url, self.space_id)
