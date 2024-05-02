from dataclasses import dataclass, field
from joblib import Memory
import pandas as pd
from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

from .base import Attribute
from ...commons import logger
from ...locations import BoundingBox, Location
if TYPE_CHECKING:
    from ...volumes import volume
    from .meta_attributes import PolylineDataAttribute
from ...retrieval import CACHE

_m = Memory(CACHE.folder, verbose=False)

@dataclass
class DataAttribute(Attribute):
    schema: str = "siibra/attr/data"

    @property
    def data(self):
        return None

    def plot(self):
        raise NotImplementedError(
            f"Plotting not implemented for {self.__class__.__name__}"
        )


BIGBRAIN_VOLUMETRIC_SHRINKAGE_FACTOR = 1.931

@dataclass
class TabularDataAttribute(DataAttribute):

    schema: str = "siibra/attr/data/tabular"
    format: str = None
    url: str = None
    plot_options: dict = field(default_factory=dict)
    parse_options: dict = field(default_factory=dict)
    
    @staticmethod
    @_m.cache
    def _GetData(url: str, parse_options: dict):
        return pd.read_csv(url, **parse_options)
        
    @property
    def data(self) -> pd.DataFrame:
        return TabularDataAttribute._GetData(self.url, self.parse_options)

    def plot(self, *args, **kwargs):
        plot_kwargs = self.plot_options.copy()
        plot_kwargs.update(kwargs)
        return self.data.plot(*args, **plot_kwargs)



@dataclass
class LayerBoundaryDataAttribute(DataAttribute):
    schema: str = "siibra/attr/data/layerboundary"
    url: str = None
    
    @staticmethod
    @_m.cache
    def _GetPolylineAttributes(url: str):
        import requests
        import numpy as np

        from .meta_attributes import PolylineDataAttribute

        LAYERS = ("0", "I", "II", "III", "IV", "V", "VI", "WM")

        all_betweeners = ("0" if start == "0" else f"{start}_{end}" for start,end in zip(LAYERS[:-1], LAYERS[1:]))
        
        def return_segments(url):
            resp = requests.get(url)
            resp.raise_for_status()
            return resp.json().get("segments")

        def poly_srt(poly: np.ndarray):
            return poly[poly[:, 0].argsort(), :]

        with ThreadPoolExecutor() as ex:
            segments = ex.map(
                return_segments,
                (f"{url}{p}.json" for p in all_betweeners)
            )
        
        return [PolylineDataAttribute(closed=False,
                                      coordinates=poly_srt(np.array(s)).tolist()
                                      ) for s in segments]
        

    @property
    def layers(self) -> list['PolylineDataAttribute']:
        return LayerBoundaryDataAttribute._GetPolylineAttributes(self.url)


@dataclass
class VoiDataAttribute(DataAttribute):

    schema: str = "siibra/attr/data/voi"
    providers: dict[str, str] = field(default_factory=dict)
    space_id: str = None

    @staticmethod
    @_m.cache
    def _GetBoundingBox(providers: dict[str, str], clip: bool=False):
        from ...volumes.providers import VolumeProvider
        for provider in providers:
            try:
                volume_provider = VolumeProvider._SUBCLASSES[provider](url=providers[provider])
                bbox = volume_provider.get_boundingbox(clip)
                return (
                    [p for p in bbox.minpoint],
                    [p for p in bbox.maxpoint],
                    bbox.space and bbox.space.id,
                )
            except Exception as e:
                logger.warn(f"Failed to get boundingbox for {provider} with url {providers[provider]}: {str(e)}")
        raise RuntimeError

    @property
    def boundingbox(self) -> "BoundingBox":
        minp, maxp, space_id = VoiDataAttribute._GetBoundingBox(self.providers)
        from ...locations import BoundingBox
        return BoundingBox(minp, maxp, space_id or self.space_id)

    @property
    def data(self) -> "volume.Volume":
        from ...volumes import volume, providers
        
        return volume.Volume(
            space_spec={"@id": self.space_id},
            providers={
                provider: providers.VolumeProvider._SUBCLASSES[provider](url=self.providers[provider])
                for provider in self.providers
            },
        )

    def plot(self, *args, **kwargs):
        raise NotImplementedError
    
    def matches(self, first_arg=None, *args, **kwargs):
        if isinstance(first_arg, Location) and first_arg.intersects(self.boundingbox):
            return True
        return super().matches(first_arg, *args, **kwargs)
