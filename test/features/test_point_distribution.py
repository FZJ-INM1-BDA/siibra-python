"""Unit tests for building PointDistribution features from specs, with local TSV files."""

import json
from unittest.mock import patch

import pandas as pd
import pytest

from siibra.configuration.factory import Factory
from siibra.core import parcellation
from siibra.core.assignment import Qualification
from siibra.core.region import Region
from siibra.features.geometric import PointDistribution
from siibra.retrieval.cache import CACHE

HOC1_LEFT, HOC1_RIGHT = Region("hOc1 left"), Region("hOc1 right")
HOC1 = Region("hOc1", children=[HOC1_LEFT, HOC1_RIGHT])
ROOT = Region("point distribution test parcellation", children=[HOC1])


class FakeParcellation:
    name = ROOT.name

    def get_region(self, name):
        matches = [r for r in ROOT.descendants if r.name == name]
        if len(matches) != 1:
            raise ValueError(f"no region {name}")
        return matches[0]


@pytest.fixture(autouse=True)
def fake_parcellation():
    with patch.object(parcellation.Parcellation, "get_instance", return_value=FakeParcellation()) as mock:
        yield mock


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    folder = tmp_path / "cache"
    folder.mkdir()
    monkeypatch.setattr(CACHE, "folder", str(folder))


@pytest.fixture
def spec(tmp_path):
    files = {}
    for name, x in [("hOc1 left", 1.0), ("hOc1 right", -1.0)]:
        path = tmp_path / f"sub-01_{name.replace(' ', '-')}.tsv"
        pd.DataFrame({"x": [x], "y": [0.0], "z": [0.0], "area_um2": [12.5]}).to_csv(path, sep="\t", index=False)
        files[name] = str(path)
    return {
        "@type": "siibra/feature/pointdistribution/v0.1",
        "name": "Cell distribution",
        "modality": "Cell body positions",
        "species": "Homo sapiens",
        "subject": "sub-01",
        "space": {"@id": "test-space"},
        "parcellation": {"@id": "test-parcellation"},
        "providers": {"tsv": files},
    }


def test_spec_matches_the_schema(spec, monkeypatch):
    jsonschema = pytest.importorskip("jsonschema")
    monkeypatch.syspath_prepend("config_schema")
    from check_schema import get_ref
    with open("config_schema/siibra/feature/pointdistribution/v0.1.json") as f:
        schema = json.load(f)
    jsonschema.validate(spec, schema, resolver=get_ref(schema))


def test_builds_one_feature_per_subject(spec):
    feature = Factory.from_json(spec)
    assert isinstance(feature, PointDistribution)
    assert feature.subject == "sub-01"
    assert feature.name == "Cell distribution - subject sub-01 (Cell body positions)"
    assert feature.fragments == ["hOc1 left", "hOc1 right"]


def test_fragment_names_are_region_names(spec):
    feature = Factory.from_json(spec)
    assert feature.anchor.regions == {HOC1_LEFT: Qualification.EXACT, HOC1_RIGHT: Qualification.EXACT}
    assert feature.matches(HOC1)


def test_fetch_points_of_a_region(spec):
    feature = Factory.from_json(spec)
    assert feature.fetch(region=HOC1_LEFT)["x"].tolist() == [1.0]
    assert feature.fetch(region="hOc1")["fragment"].tolist() == ["hOc1 left", "hOc1 right"]


def test_explicit_links_override_fragment_names(spec):
    spec["providers"] = {"tsv": {"left patch": spec["providers"]["tsv"]["hOc1 left"]}}
    spec["fragment_regions"] = {"left patch": "hOc1 left"}
    feature = Factory.from_json(spec)
    assert feature.get_fragments(HOC1) == ["left patch"]


def test_single_file_is_anchored_to_the_parcellation(spec):
    spec["providers"] = {"tsv": spec["providers"]["tsv"]["hOc1 left"]}
    feature = Factory.from_json(spec)
    assert feature.anchor._regionspec == "test-parcellation"
    assert len(feature.fetch()) == 1
