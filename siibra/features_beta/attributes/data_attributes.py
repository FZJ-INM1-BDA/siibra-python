from dataclasses import dataclass

from .base import Attribute
from ...retrieval import requests


@dataclass
class DataAttribute(Attribute, schema="siibra/attr/data"):

    def get_data(self):
        return None


@dataclass
class TabularDataAttribute(DataAttribute, schema="siibra/attr/data/tabular"):
    format: str
    url: str

    def get_data(self):
        return requests.HttpRequest(self.url).get()
