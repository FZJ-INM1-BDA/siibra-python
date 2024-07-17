from dataclasses import dataclass
from typing import TypedDict
import requests
# from io import BytesIO

from ...attributes import Attribute
from ...cache import fn_call_cache
from ...retrieval_new.file_fetcher import ZipRepository, TarRepository


class Archive(TypedDict):
    file: str = None
    format: str = None


@fn_call_cache
def get_bytesio_from_url(url: str, archive_options: Archive = None) -> bytes:
    # TODO: stream bytesio instead
    if not archive_options:
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.content

    if archive_options["format"] == "zip":
        ArchiveRepoType = ZipRepository
    elif archive_options["format"] == "tar":
        ArchiveRepoType = TarRepository
    else:
        raise NotImplementedError(f"{archive_options['format']} is not a supported archive format yet.")

    filename = archive_options["file"]
    assert filename, "Data attribute 'file' field not populated!"
    repo = ArchiveRepoType(url)
    return repo.get(filename)


@dataclass
class Data(Attribute):
    schema: str = "siibra/attr/data"
    key: str = None
    url: str = None
    archive_options: Archive = None

    def get_data(self) -> bytes:
        """
        If the data is provided in an archived format, it is decoded using the
        otherwise bytes are returned without additional steps. This is so that
        the subclasses do not need to implement their own.

        Usage
        -----
        For subclasses, call super().get_data() -> bytes
        """
        return get_bytesio_from_url(self.url, self.archive_options)
