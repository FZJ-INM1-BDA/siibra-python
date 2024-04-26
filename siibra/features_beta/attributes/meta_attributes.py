from dataclasses import dataclass

from .base import Attribute


@dataclass
class MetaAttribute(Attribute):
    type = "attr/meta"


@dataclass
class ModalityAttribute(MetaAttribute):
    type = "attr/meta/modality"
    key = "modality"
    value: str

    def filter(self, *args, modality=None, **kwargs):
        if isinstance(modality, str):
            return modality.lower() in self.value.lower()
        return super().filter(*args, modality=modality, **kwargs)


@dataclass
class NameAttribute(MetaAttribute):
    type = "attr/meta/name"
    key = "name"
    value: str

    def filter(self, *args, name=None, **kwargs):
        if isinstance(name, str):
            return name.lower() in self.value.lower()
        return super().filter(*args, name=name, **kwargs)


__all__ = [
    "ModalityAttribute",
    "NameAttribute",
]
