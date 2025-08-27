import pytest
from typing import Dict
import json
from pathlib import Path
import shutil
import siibra
from nibabel.nifti1 import Nifti1Image
from pandas import DataFrame

CUSTOM_CONF_FOLDER = "./custom-configurations/"
configuration = siibra.configuration.configuration.Configuration()


def teardown_module():
    configuration.CONFIG_EXTENSIONS = []
    shutil.rmtree(CUSTOM_CONF_FOLDER)
    siibra.cache.clear()
    siibra.use_configuration(configuration.CONFIG_CONNECTORS[0])


def create_json(conf: Dict) -> str:
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

    print(filepath.as_posix())
    return filepath.as_posix()


generic_feature_configs = [
    {
        "config": {
            "@type": "siibra/feature/tabular/v0.1",
            "name": "bla at ACBL",
            "region": "ACBL",
            "modality": "any modality",
            "description": "this describes the feature",
            "species": "Homo sapiens",
            "file": "ttps://data-proxy.ebrains.eu/api/v1/buckets/p22717-hbp-d000045_receptors-human-7A_pub/v1.0/7A_pr_examples/5-HT1A/7A_pr_5-HT1A.tsv",
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
            "file": "ttps://data-proxy.ebrains.eu/api/v1/buckets/p22717-hbp-d000045_receptors-human-7A_pub/v1.0/7A_pr_examples/5-HT1A/7A_pr_5-HT1A.tsv",
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
            "space": {"@id": "minds/core/referencespace/v1.0.0/265d32a0-3d84-40a5-926f-bf89f68212b9"},
            "providers": {
                "neuroglancer/precomputed": "https://1um.brainatlas.eu/registered_sections/bigbrain/B20_0102/precomputed"
            },
        },
        "queries": [
            ('Allen Mouse', siibra.features.generic.Image),
        ],
    },
]
conf_json_paths = []
for conf in generic_feature_configs:
    conf_json_paths.append(create_json(conf["config"]))


@pytest.mark.parametrize("conf_path", conf_json_paths)
def test_digestion(conf_path: str):
    with open(conf_path, 'rt') as fp:
        conf = json.load(fp)
    f = siibra.from_json(conf_path)
    assert f.category == "generic"
    if conf["@type"] == "siibra/feature/image/v0.1":
        assert isinstance(f, siibra.features.generic.Image)
        assert isinstance(f.fetch(), (Nifti1Image, dict))
    elif conf["@type"] == "siibra/feature/tabular/v0.1":
        assert isinstance(f, siibra.features.generic.Tabular)
        assert isinstance(f.data, DataFrame)
    else:
        raise ValueError(f'type {conf["@type"]} does not match any predefined generic types for testing.')


class TestCustomConfig:
    def setup_method(self, method):
        siibra.extend_configuration(CUSTOM_CONF_FOLDER)

    def teardown_method(self, method):
        configuration.CONFIG_EXTENSIONS = []

    @pytest.fixture(
        scope="class",
        params=[q for qs in generic_feature_configs for q in qs["queries"]],
    )
    def query_params(cls, request):
        return request.param

    def test_generic_feature_query(self, query_params):
        query_concept, query_type = query_params
        if isinstance(query_concept, str):
            query_concept = siibra.spaces.get(query_concept)
        fts = [
            f
            for f in siibra.features.get(query_concept, query_type)
            if f.category == 'generic'
        ]
        assert len(fts) > 0
        print(len(fts))
