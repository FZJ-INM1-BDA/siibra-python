from . import attributes

from ..core.structure import AnatomicalStructure
from ..configuration import Configuration

from dataclasses import dataclass, field
from typing import List
import uuid


@dataclass
class DataFeature:
    """ A multimodal data feature characterized by a set of attributes.
    """
    schema = "siibra/feature/v0.2"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attributes: List["attributes.Attribute"] = field(default_factory=list)

    @property
    def name(self):
        """ Construct name from attributes.
        TODO this needs to be more flexible.
        TODO We should allow optional specification of the name in the feature spec.
        """
        NAME_ATTRS = [
            "siibra/attr/meta/modality",
            "siibra/attr/meta/regionspec",
            "siibra/attr/meta/species",
        ]
        parts = [a.name for a in self.attributes if a.schema in NAME_ATTRS]
        return " | ".join(parts) if len(parts) > 0 else "Unnamed"

    def __post_init__(self):
        """ Construct nested FeatureAttribute objects from their specs. """
        for i, att in enumerate(self.attributes):
            if isinstance(att, DataFeature):
                continue
            elif isinstance(att, dict):
                if att.get('@type') in attributes.SCHEMAS:
                    self.attributes[i] = attributes.SCHEMAS[att.pop('@type')](**att)
                else:
                    raise RuntimeError(f"Cannot parse attribute specification type '{att.get('@type')}'")
            else:
                raise RuntimeError(f"Expecting a dictionary as feature attribute specification, not '{type(att)}'")

    def matches(self, *args, **kwargs):
        """ Returns true if this feature or one of its attributes match any of the given arguments. 
        TODO One might prefer if this matches agains **all** instead of **any** arguments, but *any* is simpler at this stage.
        """
        return any(a.matches(*args, **kwargs) for a in self.attributes)

    def get_data(self):
        return (
            attr.get_data()
            for attr in self.attributes
            if isinstance(attr, attributes.DataAttribute)
        )


def get(structure: AnatomicalStructure, modality: str, **kwargs):
    """ Query all features of a certain modality with an anatomical structure. """
    cfg = Configuration()
    return list(
        filter(
            lambda f: f.matches(modality=modality) and f.matches(structure),
            (DataFeature(**s) for _, s in cfg.specs.get("siibra/feature/v0.2"))
        )
    )
