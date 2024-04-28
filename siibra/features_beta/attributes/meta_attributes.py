from dataclasses import dataclass
from typing import Dict

from .base import Attribute

from ...core import region
from ... import commons


def all_words_in_string(s1: str, s2: str):
    """ returns true if all words from string s1 are found in  string s2. """
    return all(w in s2.lower() for w in s1.lower())


@dataclass
class MetaAttribute(Attribute, schema="siibra/attr/meta"):
    pass


@dataclass
class ModalityAttribute(MetaAttribute, schema="siibra/attr/meta/modality"):
    name: str

    def matches(self, *args, modality=None, **kwargs):
        if isinstance(modality, str) and all_words_in_string(modality, self.name):
            return True
        return super().matches(*args, modality=modality, **kwargs)


@dataclass
class AuthorAttribute(MetaAttribute, schema="siibra/attr/meta/author"):
    name: str
    affiliation: str = ""

    def matches(self, *args, name=None, affiliation=None, **kwargs):
        if isinstance(name, str) and all_words_in_string(self.name, name):
            return True
        if isinstance(affiliation, str) and all_words_in_string(self.affiliation, affiliation):
            return True
        return super().matches(*args, name=name, affiliation=affiliation, **kwargs)


@dataclass
class DoiAttribute(Attribute, schema="siibra/attr/meta/doi"):
    url: str


@dataclass
class RegionSpecAttribute(Attribute, schema="siibra/attr/meta/regionspec"):
    name: str

    def matches(self, *args, **kwargs):
        for obj in args:
            if isinstance(obj, region.Region) and obj.matches(self.name.lower()):
                return True
        return super().matches(*args, **kwargs)


@dataclass
class SpeciesSpecAttribute(Attribute, schema="siibra/attr/meta/speciesspec"):
    name: str

    def matches(self, *args, species=None, **kwargs):
        if isinstance(species, str) and all_words_in_string(self.species, species):
            return True
        if isinstance(species, commons.Species):
            raise NotImplementedError("species check not yet implemented")
        return super().matches(*args, species=species, **kwargs)


@dataclass
class EbrainsAttribute(Attribute, schema="siibra/attr/meta/ebrains"):
    ids: Dict[str, str]


__all__ = [T.__name__ for T in MetaAttribute.__subclasses__()]
