from dataclasses import dataclass
from typing import Literal

from .base import Data
from ..commons import logger
from ..cache import fn_call_cache

@dataclass
class Image(Data):

    schema: str = "siibra/attr/data/image/v0.1"
    format: Literal['nii', 'neuroglancer/precomputed'] = None
    fetcher: str = None
    space_id: str = None

    @staticmethod
    @fn_call_cache
    def _GetBoundingBox(providers: dict[str, str], clip: bool = False):
        from ..volumes.providers import VolumeProvider

        for provider in providers:
            try:
                volume_provider = VolumeProvider._SUBCLASSES[provider](
                    url=providers[provider]
                )
                bbox = volume_provider.get_boundingbox(clip)
                return (
                    [p for p in bbox.minpoint],
                    [p for p in bbox.maxpoint],
                    bbox.space and bbox.space.id,
                )
            except Exception as e:
                logger.warn(
                    f"Failed to get boundingbox for {provider} with url {providers[provider]}: {str(e)}"
                )
        raise RuntimeError

    @property
    def boundingbox(self):
        minp, maxp, space_id = Image._GetBoundingBox(self.providers)
        from ..locations import BoundingBox

        return BoundingBox(minp, maxp, space_id or self.space_id)

    @property
    def data(self):
        from ..volumes import volume, providers

        return volume.Volume(
            space_spec={"@id": self.space_id},
            providers={
                provider: providers.VolumeProvider._SUBCLASSES[provider](
                    url=self.providers[provider]
                )
                for provider in self.providers
            },
        )

    def plot(self, *args, **kwargs):
        raise NotImplementedError
