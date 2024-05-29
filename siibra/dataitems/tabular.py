from dataclasses import dataclass, field
from typing import Literal
import pandas as pd
from io import BytesIO

from .base import Data

X_DATA = "x-siibra/data/dataframe"


@dataclass
class Tabular(Data):

    schema: str = "siibra/attr/data/tabular/v0.1"
    format: Literal["csv"] = None
    plot_options: dict = field(default_factory=dict)
    parse_options: dict = field(default_factory=dict)

    def get_data(self) -> pd.DataFrame:
        _bytes = super().get_data()
        if _bytes:
            return pd.read_csv(BytesIO(_bytes), **self.parse_options)
        return pd.read_csv(self.url, **self.parse_options)

    def plot(self, *args, **kwargs):
        plot_kwargs = self.plot_options.copy()
        plot_kwargs.update(kwargs)
        return self.get_data().plot(*args, **plot_kwargs)
