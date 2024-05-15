from dataclasses import dataclass, field
from typing import Dict
import pandas as pd

from .base import Data
from ..cache import fn_call_cache

@dataclass
class Tabular(Data):

    schema: str = "siibra/attr/data/tabular"
    format: str = None
    url: str = None
    plot_options: dict = field(default_factory=dict)
    parse_options: dict = field(default_factory=dict)

    @staticmethod
    @fn_call_cache
    def _GetData(url: str, parse_options: dict):
        return pd.read_csv(url, **parse_options)

    @property
    def data(self) -> pd.DataFrame:
        return Tabular._GetData(self.url, self.parse_options)

    def plot(self, *args, **kwargs):
        plot_kwargs = self.plot_options.copy()
        plot_kwargs.update(kwargs)
        return self.data.plot(*args, **plot_kwargs)
