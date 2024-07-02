from .base import Description
from ..vocabularies import GENE_NAMES


class Gene(Description):
    schema = "siibra/desc/gene/v0.1"
    value: str = None


# WIP, currently returns list of dict, should return list of dataclass
vocab = GENE_NAMES
