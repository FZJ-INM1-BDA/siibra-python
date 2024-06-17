from dataclasses import dataclass
from typing import TypedDict, Union

from ..concepts.attribute import Attribute


class Archive(TypedDict):
    label: int = None
    file: str = None
    format: str = None


@dataclass
class Data(Attribute):
    schema: str = "siibra/attr/data"
    key: str = None
    url: str = None
    archive_options: Archive = None

    def get_data(self) -> Union[bytes, None]:
        """
        If the data is provided in an archived format, it is decoded using the
        Data.get_data method. This is so that the subclasses do not need to implement their own

        Usage
        -----
        For subclasses, call super().get_data() -> Union[bytes, None]. Catch the scenarios where
        None is returned (which implies the data is not in an archive)
        """
        if self.archive_options and self.archive_options["format"] == "zip":

            from ..retrieval_new.file_fetcher import ZipRepository

            filename = self.archive_options["file"]
            assert filename, "Data attribute 'file' field not populated!"
            repo = ZipRepository(self.url)
            return repo.get(filename)
