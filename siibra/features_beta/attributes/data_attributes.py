from dataclasses import dataclass, field
from joblib import Memory
import pandas as pd
from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

from .base import Attribute
from ...commons import logger
from ...locations import BoundingBox, Location

if TYPE_CHECKING:
    from ...volumes import volume
    from .meta_attributes import PolylineDataAttribute


@dataclass
class DataAttribute(Attribute):
    schema: str = "siibra/attr/data"

    @property
    def data(self):
        return None

    def plot(self):
        raise NotImplementedError(
            f"Plotting not implemented for {self.__class__.__name__}"
        )


BIGBRAIN_VOLUMETRIC_SHRINKAGE_FACTOR = 1.931

