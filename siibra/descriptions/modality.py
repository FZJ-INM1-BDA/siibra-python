from dataclasses import dataclass

from ..concepts import attribute
from .base import Description

@dataclass
class Modality(Description):
    schema = "siibra/attr/desc/modality"
    name: str = None

    # @staticmethod
    # def _GetAll():
    #     from ...configuration import Configuration
    #     from ..feature import DataFeature

    #     cfg = Configuration()
    #     return {
    #         attr.name
    #         for _, s in cfg.specs.get("siibra/feature/v0.2")
    #         for attr in DataFeature(**s).attributes
    #         if isinstance(attr, ModalityAttribute)
    #     }

    # def matches(self, *args, modality=None, **kwargs):
    #     if isinstance(modality, str) and all_words_in_string(modality, self.name):
    #         return True
    #     return super().matches(*args, modality=modality, **kwargs)

