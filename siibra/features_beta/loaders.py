from functools import wraps
from typing import Iterable, List, Callable

from .feature import Feature
from .attributes import NameAttribute, ModalityAttribute

from siibra.configuration import Configuration
from siibra import __version__
from siibra.retrieval.repositories import GitlabConnector, RepositoryConnector

SERVER = "https://jugit.fz-juelich.de"
PROJECT = "3484"

connector = GitlabConnector(SERVER, PROJECT, f"siibra-{__version__}",
                            skip_branchtest=True)

FEATURE_ITERATORS: List[Callable[..., Iterable[Feature]]] = []

def get_instances():
    for c in FEATURE_ITERATORS:
        yield from c()

def parse_conf_folder(conf_folder):
    def outer(fn):
        files = connector.search_files(conf_folder, suffix=".json")
        FEATURE_ITERATORS.append(lambda: (fn(connector.get(file)) for file in files))
    return outer

@parse_conf_folder("features/tabular/corticalprofiles/receptor")
def process_receptor_density_profiles(json_obj):
    assert json_obj.get("@type") == "siibra/feature/profile/receptor/v0.1"
    receptor = json_obj.get("receptor")
    return Feature(
        name=f"Receptor Density Profile of {receptor}",
        id="",
        desc="",
        attributes=[
            NameAttribute(f"Receptor Density Profile of {receptor}"),
            ModalityAttribute("molecular/ReceptorDensityProfile")
        ]
    )
    
    