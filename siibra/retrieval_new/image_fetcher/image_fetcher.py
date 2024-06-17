from abc import ABC, abstractmethod
from typing import Type, TYPE_CHECKING, TypedDict
from pathlib import Path
from typing import Union
import requests

import nibabel as nib

from ...cache import CACHE
from ..file_fetcher import ZipRepository, TarRepository
from ...commons import SIIBRA_MAX_FETCH_SIZE_GIB

if TYPE_CHECKING:
    from ...locations import BBox
    from ...dataitems import Archive


def cache_and_load_img(
    url: str, archive_options: "Archive" = None
) -> Union[nib.Nifti1Image, nib.Nifti2Image]:
    if archive_options:
        if archive_options["format"] == "zip":
            ArchiveRepoType = ZipRepository
        elif archive_options["format"] == "tar":
            ArchiveRepoType = TarRepository
        fname = archive_options["file"]
        suffix = "".join(Path(fname).suffixes)
        repo = ArchiveRepoType(url)
        cache_fname = CACHE.build_filename(url + fname, suffix=suffix)
        if not Path(cache_fname).exists():
            with open(cache_fname, "wb") as fp:
                fp.write(repo.get(fname))
    else:
        suffix = "".join(Path(url).suffixes)
        cache_fname = CACHE.build_filename(url, suffix=suffix)
        if not Path(cache_fname).exists():
            with open(cache_fname, "wb") as fp:
                resp = requests.get(url)
                resp.raise_for_status()
                fp.write(resp.content)
    nii = nib.load(cache_fname)
    return nii


class FetchKwargs(TypedDict):
    """
    Key word arguments used for fetching images and meshes across siibra.
    """

    bbox: "BBox" = (None,)
    resolution_mm: float = (None,)
    max_download_GB: float = (SIIBRA_MAX_FETCH_SIZE_GIB,)
    color_channel: int = None


class ImageFetcher(ABC):

    srcformat: str = None
    SUBCLASSES: dict[str, Type["ImageFetcher"]] = {}

    def __init__(self, url):
        self.url = url

    def __init_subclass__(cls, srcformat: str) -> None:
        assert (
            srcformat not in ImageFetcher.SUBCLASSES
        ), f"{srcformat} already registered."
        cls.srcformat = srcformat
        ImageFetcher.SUBCLASSES[srcformat] = cls
        return super().__init_subclass__()

    # @abstractmethod
    # def get_bbox(self, clip=True, background=0.0) -> "BBox":
    #     raise NotImplementedError

    @abstractmethod
    def fetch(self, *args, **kwargs):
        raise NotImplementedError
