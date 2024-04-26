from ..core.structure import AnatomicalStructure

from dataclasses import dataclass
from typing import List, Iterable

import attributes


@dataclass
class Feature:
    name: str
    desc: str
    id: str
    attributes: List["attributes.Attribute"]

    @property
    def modalities(self):
        return get_feature_modalities(self)


def get_feature_modalities(feature: Feature) -> list[str]:
    return [
        attr.modality
        for attr in feature.attributes
        if isinstance(attr, attributes.ModalityAttribute)
    ]


def get(structure: AnatomicalStructure, modality: str, **kwargs):
    def match_filter(f):
        return (
            modality in get_feature_modalities(f)
            and all(attr.match(**kwargs) for attr in f.attributes)
            and structure.matches(f.anchor)
        )

    return filter(match_filter, get_all_features())


def get_all_features() -> Iterable['Feature']:
    yield []
