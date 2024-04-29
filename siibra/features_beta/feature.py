from . import attributes

from ..core.structure import AnatomicalStructure
from ..configuration import Configuration

from dataclasses import dataclass, field
from typing import List
import uuid


NAME_ATTRS = [
    "siibra/attr/meta/modality",
    "siibra/attr/meta/regionspec",
    "siibra/attr/meta/species",
]


@dataclass
class DataFeature:
    """ A multimodal data feature characterized by a set of attributes.
    """
    schema = "siibra/feature/v0.2"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attributes: List["attributes.Attribute"] = field(default_factory=list)
    name: str = field(default=None)

    def __post_init__(self):
        # Construct nested FeatureAttribute objects from their specs
        parsed_attrs = []
        for att in self.attributes:
            if isinstance(att, DataFeature):
                continue
            elif isinstance(att, dict):
                if att.get('@type') in attributes.SCHEMAS:
                    parsed_attrs.append(
                        attributes.SCHEMAS[att.pop('@type')](**att)
                    )
                else:
                    raise RuntimeError(f"Cannot parse attribute specification type '{att.get('@type')}'")
            else:
                raise RuntimeError(f"Expecting a dictionary as feature attribute specification, not '{type(att)}'")
        self.attributes = parsed_attrs

        if self.name is None:
            parts = [a.name for a in self.attributes if a.schema in NAME_ATTRS]
            self.name = ", ".join(parts) if len(parts) > 0 else "Unnamed"

    def matches(self, *args, **kwargs):
        """ Returns true if this feature or one of its attributes match any of the given arguments.
        TODO One might prefer if this matches agains **all** instead of **any** arguments, but *any* is simpler at this stage.
        """
        return any(a.matches(*args, **kwargs) for a in self.attributes)

    def get_data(self):
        """ Return a list data obtained from DataAttributes.
        TODO this is just for beta development, later on we might rather
        have properly typed methods to load any available data frames and images.
        """
        return (
            attr.data
            for attr in self.attributes
            if isinstance(attr, attributes.DataAttribute)
        )

    def plot(self, *args, **kwargs):
        """ Plots all data attributes.
        """
        for attr in self.attributes:
            if isinstance(attr, attributes.DataAttribute):
                attr.plot(*args, **kwargs)


def get(structure: AnatomicalStructure, modality: str, **kwargs):
    """ Query all features of a certain modality with an anatomical structure. """
    cfg = Configuration()
    return list(
        filter(
            lambda f: f.matches(modality=modality) and f.matches(region=structure), # Ideally enforce only keyword arguement
            (DataFeature(**s) for _, s in cfg.specs.get("siibra/feature/v0.2"))
        )
    )
