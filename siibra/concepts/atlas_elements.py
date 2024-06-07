from dataclasses import dataclass

from .attribute_collection import AttributeCollection
from ..descriptions import (
    Name,
    SpeciesSpec,
    Url,
    Doi,
    ID as _ID,
    TextDescription,
    Modality,
)

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
    def species(self):
        species = [s.value for s in self._finditer(SpeciesSpec)]
        assert len(species) == 1
        return species[0]

    @property
    def publications(self):

        from ..retrieval_new.doi_fetcher import get_citation

        citations = [
            Url(value=doi.value, text=get_citation(doi)) for doi in self._find(Doi)
        ]

        return [*self._find(Url), *citations]

    @property
    def description(self):
        return self._get(TextDescription).value

    @property
    def modalities(self):
        return [mod.value for mod in self._find(Modality)]
