from dataclasses import dataclass
from typing import List

from .attributes import Attribute


@dataclass
class Feature:
    name: str
    desc: str
    id: str
    attributes: List[Attribute]

    def filter(self, *args, **kwargs):
        for attr in self.attributes:
            if not attr.filter(*args, **kwargs):
                return False
        return True
