from dataclasses import dataclass, field

from .base import Attribute
from ...commons import logger
from ...retrieval import requests


@dataclass
class DataAttribute(Attribute, schema="siibra/attr/data"):

    @property
    def data(self):
        return None

    def plot(self):
        raise NotImplementedError(
            f"Plotting not implemented for {self.__class__.__name__}"
        )


@dataclass
class TabularDataAttribute(DataAttribute, schema="siibra/attr/data/tabular"):
    format: str
    url: str
    index_column: int = None
    plot_options: dict = field(default_factory=dict)

    @property
    def data(self):
        df = requests.HttpRequest(self.url).get()
        if isinstance(self.index_column, int):
            try:
                df.set_index(df.columns[self.index_column], inplace=True)
            except IndexError:
                logger.warn(f"Could not set index to #{self.index_column} of columns {df.columns}.")
        return df

    def plot(self, *args, **kwargs):
        plot_kwargs = self.plot_options.copy()
        plot_kwargs.update(kwargs)
        return self.data.plot(*args, **plot_kwargs)
