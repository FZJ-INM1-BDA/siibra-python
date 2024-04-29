from dataclasses import dataclass, field
from joblib import Memory
from pandas import DataFrame
from typing import TYPE_CHECKING

from .base import Attribute
from ...commons import logger
if TYPE_CHECKING:
    from ...volumes import volume
    from ...locations import BoundingBox
from ...retrieval import requests, CACHE

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


@dataclass
class TabularDataAttribute(DataAttribute):

    schema: str = "siibra/attr/data/tabular"
    format: str = None
    url: str = None
    index_column: int = None
    plot_options: dict = field(default_factory=dict)
    

    @staticmethod
    @_m.cache
    def _GetData(url: str, index_column: int):
        df = requests.HttpRequest(url).get()
        assert isinstance(df, DataFrame), f"Expected tabular data to be dataframe, but is {type(df)} instead"
        if isinstance(index_column, int):
            try:
                df.set_index(df.columns[index_column], inplace=True)
            except IndexError:
                logger.warn(f"Could not set index to #{index_column} of columns {df.columns}.")
        return df

    @property
    def data(self) -> DataFrame:
        return TabularDataAttribute._GetData(self.url, self.index_column)

    def plot(self, *args, **kwargs):
        plot_kwargs = self.plot_options.copy()
        plot_kwargs.update(kwargs)
        return self.data.plot(*args, **plot_kwargs)


@dataclass
class VoiDataAttribute(DataAttribute):

    schema: str = "siibra/attr/data/voi"
    providers: dict[str, str] = field(default_factory=dict)
    space: str = None

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
        return BoundingBox(minp, maxp, space_id or self.space)

    @property
    def data(self) -> "volume.Volume":
        from ...volumes import volume, providers
        
        return volume.Volume(
            space_spec={"@id": self.space},
            providers={
                provider: providers.VolumeProvider._SUBCLASSES[provider](url=self.providers[provider])
                for provider in self.providers
            },
        )

    def plot(self, *args, **kwargs):
        raise NotImplementedError
    
    def matches(self, *args, bbox: "BoundingBox"=None, **kwargs):
        if bbox and bbox.intersects(self.boundingbox):
            return True
        return super().matches(*args, **kwargs)
