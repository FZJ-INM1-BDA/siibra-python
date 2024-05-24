from dataclasses import dataclass

from .attribute_collection import AttributeCollection
from ..descriptions import Name, SpeciesSpec, Url, Doi, ID as _ID


MUSTHAVE_ATTRIBUTES = {Name, _ID, SpeciesSpec}


@dataclass
class AtlasElement(AttributeCollection):
    schema: str = "siibra/atlas_element/v0.1"

    def __post_init__(self):
        attr_types = set(map(type, self.attributes))
        assert all(
            musthave in attr_types for musthave in MUSTHAVE_ATTRIBUTES
        ), f"An AtlasElement must have {[attr_type.__name__ for attr_type in MUSTHAVE_ATTRIBUTES]} attributes."

    @property
    def name(self):
        names = [s.value for s in self.getiter(Name)]
        assert len(names) == 1
        return names[0]

    @property
    def ID(self):
        ids = [id.value for id in self.getiter(_ID)]
        assert len(ids) == 1
        return ids[0]

    @property
    def species(self):
        species = [s.value for s in self.getiter(SpeciesSpec)]
        assert len(species) == 1
        return species[0]

    @property
    def publications(self):
        return self.get(Url) + self.get(Doi)
