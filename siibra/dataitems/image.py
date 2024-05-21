from dataclasses import dataclass
from typing import Literal, Union
import requests
import nibabel as nib
from pathlib import Path

from .base import Data
from ..commons import logger
from ..cache import CACHE, fn_call_cache

@dataclass
class Image(Data):

    schema: str = "siibra/attr/data/image/v0.1"
    format: Literal['nii', 'neuroglancer/precomputed'] = None
    fetcher: str = None
    key: str = None
    space_id: str = None

    @staticmethod
    @fn_call_cache
    def _GetBoundingBox(providers: dict[str, str], clip: bool = False):
        # deprecated
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
        # deprecated
        minp, maxp, space_id = Image._GetBoundingBox(self.providers)
        from ..locations import BoundingBox

        return BoundingBox(minp, maxp, space_id or self.space_id)

    @property
    def data(self):
        assert self.format == "nii", f"Can only get data of nii."
        return Image.NiiUrl(self.fetcher)

    
    @staticmethod
    def NiiUrl(url: str) -> Union[nib.Nifti1Image, nib.Nifti2Image]:
        filename = CACHE.build_filename(url, suffix=".nii.gz")
        if not Path(filename).exists():
            with open(filename, "wb") as fp:
                resp = requests.get(url)
                resp.raise_for_status()
                fp.write(resp.content)
        nii = nib.load(filename)
        return nii

    def plot(self, *args, **kwargs):
        raise NotImplementedError
