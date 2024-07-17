from dataclasses import dataclass
from packaging.version import Version as PackagingVersion

from .base import Description


@dataclass
class Version(Description):
    schema = "siibra/attr/desc/version/v0.1"
    prev_id: str = None  # None if there is no previous
    next_id: str = None  # None if there is no next

    def __lt__(self, other: "Version"):
        assert isinstance(other, Version), TypeError(f"'>' not supported between instances of '{type(self)}' and '{type(other)}'")
        return PackagingVersion(self.value) < PackagingVersion(other.value)
