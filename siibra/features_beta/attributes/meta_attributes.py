from dataclasses import dataclass, field
from typing import Dict
from joblib import Memory

from .base import Attribute

from ...core import region as _region, parcellation as _parcellation
from ...locations import Location, Point, PointSet
from ...retrieval import CACHE
from ... import commons

_m = Memory(CACHE.folder, verbose=False)


def all_words_in_string(s1: str, s2: str):
    """returns true if all words from string s1 are found in  string s2."""
    return all(w in s2.lower() for w in s1.lower())


@dataclass
class MetaAttribute(Attribute):
    schema = "siibra/attr/meta"


@dataclass
class ModalityAttribute(MetaAttribute):
    schema = "siibra/attr/meta/modality"
    name: str = None

    @staticmethod
    def _GetAll():
        from ...configuration import Configuration
        from ..feature import DataFeature

        cfg = Configuration()
        return {
            attr.name
            for _, s in cfg.specs.get("siibra/feature/v0.2")
            for attr in DataFeature(**s).attributes
            if isinstance(attr, ModalityAttribute)
        }

    def matches(self, *args, modality=None, **kwargs):
        if isinstance(modality, str) and all_words_in_string(modality, self.name):
            return True
        return super().matches(*args, modality=modality, **kwargs)


@dataclass
class AuthorAttribute(MetaAttribute):
    schema = "siibra/attr/meta/author"
    name: str = None
    affiliation: str = ""

    def matches(self, *args, name=None, affiliation=None, **kwargs):
        if isinstance(name, str) and all_words_in_string(self.name, name):
            return True
        if isinstance(affiliation, str) and all_words_in_string(
            self.affiliation, affiliation
        ):
            return True
        return super().matches(*args, name=name, affiliation=affiliation, **kwargs)


@dataclass
class DoiAttribute(MetaAttribute):
    schema = "siibra/attr/meta/doi"
    url: str = None


@dataclass
class IDAttribute(MetaAttribute):
    schema = "siibra/attr/meta/id"
    uuid: str = None


@dataclass
class RegionSpecAttribute(MetaAttribute):
    schema = "siibra/attr/meta/regionspec"
    name: str = None

    @staticmethod
    @_m.cache
    def Matches(regionspec: str, parcellation: str, region: str) -> bool:
        found_region = _parcellation.Parcellation.registry()[parcellation].get_region(
            region
        )
        return found_region.matches(regionspec)

    def matches(self, first_arg=None, *args, region=None, parcellation=None, **kwargs):
        if isinstance(first_arg, _region.Region):
            region = first_arg
        if isinstance(region, _region.Region):
            return RegionSpecAttribute.Matches(
                self.name, region.parcellation.name, region.name
            )
        if isinstance(region, str):
            assert (
                parcellation
            ), "If region is supplied as a string, parcellation must be defined!"
            assert isinstance(
                parcellation, (str, _parcellation.Parcellation)
            ), "parcellation must be of type str or Parcellation"
            if isinstance(parcellation, _parcellation.Parcellation):
                parcellation = parcellation.name
            return RegionSpecAttribute.Matches(self.name, parcellation, region)
        return super().matches(first_arg, *args, **kwargs)


@dataclass
class PointAttribute(MetaAttribute):
    schema = "siibra/attr/meta/point"
    space_id: str = None
    coordinate: list[float] = field(default_factory=list)

    def matches(self, first_arg=None, *args, location: Location = None, **kwargs):

        if isinstance(first_arg, Location):
            location = first_arg
        if location and location.intersects(
            Point(self.coordinate, space=self.space_id)
        ):
            return True
        return super().matches(first_arg, *args, **kwargs)


@dataclass
class PointSetAttribute(MetaAttribute):
    schema = "siibra/attr/meta/pointset"
    space_id: str = None
    coordinates: list[list[float]] = field(default_factory=list)

    def matches(self, first_arg=None, *args, location: Location = None, **kwargs):

        if isinstance(first_arg, Location):
            location = first_arg
        if location and location.intersects(
            PointSet(self.coordinates, space=self.space_id)
        ):
            return True
        return super().matches(first_arg, *args, **kwargs)


@dataclass
class PolylineDataAttribute(MetaAttribute):
    schema: str = "siibra/attr/meta/polyline"
    closed: bool = False
    space_id: str = None
    coordinates: list[list[float]] = field(default_factory=list)


@dataclass
class SpeciesSpecAttribute(MetaAttribute):
    schema = "siibra/attr/meta/speciesspec"
    name: str = None

    def matches(self, *args, species=None, **kwargs):
        if isinstance(species, str) and all_words_in_string(self.species, species):
            return True
        if isinstance(species, commons.Species):
            raise NotImplementedError("species check not yet implemented")
        return super().matches(*args, species=species, **kwargs)


@dataclass
class EbrainsAttribute(MetaAttribute):
    schema = "siibra/attr/meta/ebrains"
    ids: Dict[str, str] = None


__all__ = [T.__name__ for T in MetaAttribute.__subclasses__()]
