from dataclasses import dataclass, field
from joblib import Memory
from pandas import DataFrame

from .base import Attribute
from ...commons import logger
from ...retrieval import requests, CACHE

_m = Memory(CACHE.folder, verbose=False)

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


@dataclass
class TabularDataAttribute(DataAttribute):

    schema: str = "siibra/attr/data/tabular"
    format: str = None
    url: str = None
    index_column: int = None
    plot_options: dict = field(default_factory=dict)
    

    @staticmethod
    @_m.cache
    def _GetData(url: str, index_column: int):
        df = requests.HttpRequest(url).get()
        assert isinstance(df, DataFrame), f"Expected tabular data to be dataframe, but is {type(df)} instead"
        if isinstance(index_column, int):
            try:
                df.set_index(df.columns[index_column], inplace=True)
            except IndexError:
                logger.warn(f"Could not set index to #{index_column} of columns {df.columns}.")
        return df

    @property
    def data(self) -> DataFrame:
        return TabularDataAttribute._GetData(self.url, self.index_column)

    def plot(self, *args, **kwargs):
        plot_kwargs = self.plot_options.copy()
        plot_kwargs.update(kwargs)
        return self.data.plot(*args, **plot_kwargs)
