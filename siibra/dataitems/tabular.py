from dataclasses import dataclass, field
from typing import Literal, Dict, Union
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
        if X_DATA in self.extra:
            return self.extra[X_DATA]
        _bytes = super().get_data()
        if _bytes:
            return pd.read_csv(BytesIO(_bytes), **self.parse_options)
        return pd.read_csv(self.url, **self.parse_options)

    def plot(self, *args, **kwargs):
        if "scatter" in self.plot_options:
            scatter_kwargs: Dict[str, Union[str, int, float]] = self.plot_options.get("scatter").copy()
            scatter_kwargs.update(kwargs)
            return self.get_data().plot.scatter(*args, **scatter_kwargs)
        plot_kwargs = self.plot_options.copy()
        plot_kwargs.update(kwargs)
        return self.get_data().plot(*args, **plot_kwargs)
