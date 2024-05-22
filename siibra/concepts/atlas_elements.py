from dataclasses import dataclass

from .attribute_collection import AttributeCollection
from ..descriptions import Name, SpeciesSpec, Publication, ID as _ID


MUSTHAVE_ATTRIBUTES = {Name, _ID, SpeciesSpec}


@dataclass
class AtlasElement(AttributeCollection):
    schema: str = "siibra/atlas_element/v0.1"

    def __post_init__(self):
        assert all(
            any(isinstance(att, musthave) for att in self.attributes)
            for musthave in MUSTHAVE_ATTRIBUTES
        ), F"An AtlasElement must have {MUSTHAVE_ATTRIBUTES} attributes."

    @property
    def name(self):
        names = [s.value for s in self.get(Name)]
        assert len(names) == 1
        return names[0]

    @property
    def ID(self):
        return [id.value for id in self.get(_ID)]

    @property
    def species(self):
        species = [s.value for s in self.get(SpeciesSpec)]
        assert len(species) == 1
        return species[0]

    @property
    def publications(self) -> "Publication":
        return self.get(Publication)
