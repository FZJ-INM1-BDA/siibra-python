import pytest
from typing import Dict
import json
from pathlib import Path
import shutil
import siibra
from nibabel.nifti1 import Nifti1Image
from pandas import DataFrame

CUSTOM_CONF_FOLDER = "./custom-configurations/"


def create_json(conf: Dict):
    if conf["@type"] == "siibra/feature/tabular/v0.1":
        folderpath = Path(CUSTOM_CONF_FOLDER + "features/tabular/")
    elif conf["@type"] == "siibra/feature/image/v0.1":
        folderpath = Path(CUSTOM_CONF_FOLDER + "features/images/")
    else:
        raise NotImplementedError(f"There is no generic feature type '{conf['@type']}'")

    folderpath.mkdir(parents=True, exist_ok=True)
    filepath = folderpath.joinpath(conf.get("name") + ".json")
    with open(filepath, "wt") as fp:
        json.dump(conf, fp=fp)

    return filepath.as_posix()


siibra.spaces

generic_feature_configs = [
    {
        "config": {
            "@type": "siibra/feature/tabular/v0.1",
            "name": "bla at ACBL",
            "region": "ACBL",
            "modality": "any modality",
            "description": "this describes the feature",
            "species": "Homo sapiens",
            "file": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-7A_pub/v1.0/7A_pr_examples/5-HT1A/7A_pr_5-HT1A.tsv",
        },
        "queries": [
            (siibra.get_region("julich 3.1", "acbl"), siibra.features.generic.Tabular),
            (siibra.get_region("julich 3.1", "basal ganglia"), siibra.features.generic.Tabular),
            (siibra.get_region("julich 3.1", "acbl right"), siibra.features.generic.Tabular),
            (siibra.get_region("julich 3.0", "ventral striatum"), siibra.features.generic.Tabular),
        ],
    },
    {
        "config": {
            "@type": "siibra/feature/tabular/v0.1",
            "name": "Cochlear nucleus, ventral part data",
            "region": "Cochlear nucleus, ventral part",
            "modality": "any modality",
            "description": "this describes the feature",
            "species": "Rattus norvegicus",
            "file": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-7A_pub/v1.0/7A_pr_examples/5-HT1A/7A_pr_5-HT1A.tsv",
        },
        "queries": [
            (siibra.get_region("Waxholm 4", "Cochlear nucleus, ventral part"), siibra.features.generic.Tabular),
            (siibra.get_region("Waxholm 4", "Rhombencephalon"), siibra.features.generic.Tabular),
            (siibra.get_region("Waxholm 4", "Ventral cochlear nucleus, anterior part"), siibra.features.generic.Tabular),
        ],
    },
    {
        "config": {
            "@type": "siibra/feature/image/v0.1",
            "name": "some name for custom image feature",
            "modality": "foo",
            "ebrains": {"openminds/DatasetVersion": "73c1fa55-d099-4854-8cda-c9a403c6080a"},
            "space": {"@id": "minds/core/referencespace/v1.0.0/d5717c4a-0fa1-46e6-918c-b8003069ade8"},
            "providers": {
                "neuroglancer/precomputed": "https://1um.brainatlas.eu/registered_sections/bigbrain/B20_0102/precomputed"
            },
        },
        "queries": [
            ('waxholm', siibra.features.generic.Image),
            ('waxholm', siibra.features.generic.Image),
            ('waxholm', siibra.features.generic.Image),
        ],
    },
]

conf_jsons = []
for conf in generic_feature_configs:
    conf_jsons.append(create_json(conf["config"]))


@pytest.mark.parametrize("conf_path", conf_jsons)
def test_digestion(conf_path: str):
    with open(conf_path, 'rt') as fp:
        conf = json.load(fp)
    f = siibra.from_json(conf_path)
    assert f.category == 'generic'
    if conf["@type"] == "siibra/feature/image/v0.1":
        assert isinstance(f, siibra.features.generic.Image)
        assert isinstance(f.fetch(), (Nifti1Image, dict))
    elif conf["@type"] == "siibra/feature/tabular/v0.1":
        assert isinstance(f, siibra.features.generic.Tabular)
        assert isinstance(f.data, DataFrame)
    else:
        raise ValueError(f'type {conf["@type"]} does not match any predefined generic types for testing.')


queries = [q for qs in generic_feature_configs for q in qs["queries"]]


@pytest.mark.parametrize("query_concept, query_type", queries)
def test_generic_feature_query(query_concept, query_type: siibra.features.Feature):
    if isinstance(query_concept, str):
        query_concept = siibra.spaces.get(query_concept)  # TODO: check why match method fails
    fts = [
        f
        for f in siibra.features.get(query_concept, query_type)
        if isinstance(f, query_type)
    ]
    assert len(fts) > 0


@pytest.fixture(scope="module", autouse=True)
def prepare_and_cleanup_module():
    siibra.extend_configuration(CUSTOM_CONF_FOLDER)
    yield
    shutil.rmtree(CUSTOM_CONF_FOLDER)
    siibra.configuration.configuration.Configuration().CONFIG_EXTENSIONS = []
