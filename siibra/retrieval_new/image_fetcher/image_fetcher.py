from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type, TYPE_CHECKING
from pathlib import Path
from typing import Union
import requests

import nibabel as nib
from ...cache import CACHE

from ...commons import SIIBRA_MAX_FETCH_SIZE_GIB

if TYPE_CHECKING:
    from ...locations import BBox


def cache_and_load_img(url: str) -> Union[nib.Nifti1Image, nib.Nifti2Image]:
    filename = CACHE.build_filename(url, suffix=".nii.gz")
    if not Path(filename).exists():
        with open(filename, "wb") as fp:
            resp = requests.get(url)
            resp.raise_for_status()
            fp.write(resp.content)
    nii = nib.load(filename)
    return nii


@dataclass
class FetchKwargs:
    """
    Key word arguments used for fetching images and meshes across siibra.
    """
    bbox: "BBox" = None,
    resolution_mm: float = None,
    max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB,
    label: int = None,
    color_channel: int = None


class ImageFetcher(ABC):

    srcformat: str = None
    SUBCLASSES: dict[str, Type["ImageFetcher"]] = {}

    def __init__(self, url):
        self.url = url

    def __init_subclass__(cls, srcformat: str) -> None:
        assert srcformat not in ImageFetcher.SUBCLASSES, f"{srcformat} already registered."
        cls.srcformat = srcformat
        ImageFetcher.SUBCLASSES[srcformat] = cls
        return super().__init_subclass__()

    # @abstractmethod
    # def get_bbox(self, clip=True, background=0.0) -> "BBox":
    #     raise NotImplementedError

    @abstractmethod
    def fetch(self, *args, **kwargs):
        raise NotImplementedError
