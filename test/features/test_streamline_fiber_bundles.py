"""Unit tests for building StreamlineFiberBundles from specs, with a local TRX file and table."""

import json
import logging
from unittest.mock import Mock, patch

import pandas as pd
import pytest

pytest.importorskip("trx", reason="trx-python requires Python >= 3.11")

from siibra.commons import Species  # noqa: E402
from siibra.configuration.factory import Factory  # noqa: E402
from siibra.core import parcellation, space  # noqa: E402
from siibra.core.assignment import Qualification  # noqa: E402
from siibra.core.region import Region  # noqa: E402
from siibra.features.geometric import StreamlineFiberBundles  # noqa: E402
from test.geometries.tractogram_files import make_streamlines, write_trx  # noqa: E402

GROUPS = {"Left_Arcuate": [0, 1], "Left_Arcuate_anterior": [1, 2], "Right_Uncinate": [3, 4]}
HOC1_LEFT, HOC1_RIGHT, HOC2_LEFT = Region("hOc1 left"), Region("hOc1 right"), Region("hOc2 left")
HOC1 = Region("hOc1", children=[HOC1_LEFT, HOC1_RIGHT])
UNRELATED = Region("Area 44 left")
ROOT = Region("streamline bundle test parcellation", children=[HOC1, HOC2_LEFT, UNRELATED])


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


@pytest.fixture
def spec(tmp_path):
    table = pd.DataFrame(
        [["Left_Arcuate", 1, 0, 0], ["Left_Arcuate_anterior", 1, 1, 1],
         ["Right_Uncinate", 0, 0, 0], ["Extra_Bundle", 0, 1, 0]],
        columns=["bundle", "hOc1 left", "hOc2 left", "hOc9 left"],
    )
    table.to_csv(tmp_path / "terminations.tsv", sep="\t", index=False)
    return {
        "@type": "siibra/feature/streamlinefiberbundles/v0.2",
        "name": "Long fiber bundles",
        "modality": "Streamline fiber bundles",
        "species": "Homo sapiens",
        "space": {"@id": "test-space"},
        "parcellation": {"@id": "test-parcellation"},
        "providers": {"trx": write_trx(tmp_path / "bundles.trx", make_streamlines(), GROUPS)},
        "fragment_regions": str(tmp_path / "terminations.tsv"),
    }


def test_spec_matches_the_schema(spec, monkeypatch):
    jsonschema = pytest.importorskip("jsonschema")
    monkeypatch.syspath_prepend("config_schema")
    from check_schema import get_ref
    with open("config_schema/siibra/feature/streamlinefiberbundles/v0.2.json") as f:
        schema = json.load(f)
    jsonschema.validate(spec, schema, resolver=get_ref(schema))


def test_builds_from_spec(spec):
    feature = Factory.from_json(spec)
    assert isinstance(feature, StreamlineFiberBundles)
    assert feature.name == "Long fiber bundles (Streamline fiber bundles)"
    assert feature.providers == {"trx": spec["providers"]["trx"]}
    assert feature.anchor.species == {Species.decode("Homo sapiens")}


def test_anchor_holds_all_termination_regions_as_overlaps(spec):
    feature = Factory.from_json(spec)
    assert feature.anchor.regions == {HOC1_LEFT: Qualification.OVERLAPS, HOC2_LEFT: Qualification.OVERLAPS}


def test_unresolvable_region_names_are_reported(spec, caplog):
    with caplog.at_level(logging.WARNING, logger="siibra"):
        feature = Factory.from_json(spec)
    assert "hOc9 left" in caplog.text
    assert feature.fragment_regions["Left_Arcuate_anterior"] == [HOC1_LEFT, HOC2_LEFT]


def test_table_rows_without_terminations_are_kept(spec):
    assert Factory.from_json(spec).fragment_regions["Right_Uncinate"] == []


def test_found_by_regions_the_bundles_terminate_in(spec):
    feature = Factory.from_json(spec)
    assert feature.matches(HOC1_LEFT)
    assert feature.matches(HOC1)  # a parent of a termination region
    assert not feature.matches(UNRELATED)


def test_fetch_bundles_terminating_in_a_region(spec):
    feature = Factory.from_json(spec)
    assert len(feature.fetch(region=HOC1_LEFT).streamlines) == 3  # streamlines 0, 1, 2
    assert len(feature.fetch(region="hOc2 left").streamlines) == 2  # streamlines 1, 2
    assert len(feature.fetch(fragment="Right_Uncinate").streamlines) == 2


def test_table_rows_and_bundles_are_compared_on_first_access(spec, caplog):
    feature = Factory.from_json(spec)
    with caplog.at_level(logging.WARNING, logger="siibra"):
        feature.fetch(fragment="Left_Arcuate")
    assert "Extra_Bundle" in caplog.text


def test_inline_links(spec):
    spec["fragment_regions"] = {"Left_Arcuate": "hOc1 right", "Right_Uncinate": ["hOc2 left"]}
    feature = Factory.from_json(spec)
    assert feature.get_fragments(HOC1) == ["Left_Arcuate"]
    assert set(feature.anchor.regions) == {HOC1_RIGHT, HOC2_LEFT}


def test_without_links_the_tractogram_is_anchored_to_the_parcellation(spec):
    del spec["fragment_regions"]
    feature = Factory.from_json(spec)
    assert feature.anchor._regionspec == "test-parcellation"
    assert feature.fragment_regions == {}
    assert len(feature.fetch(fragment="Left_Arcuate").streamlines) == 2


def test_species_default_to_the_space(spec):
    del spec["species"]
    with patch.object(space.Space, "get_instance", return_value=Mock(species=Species.decode("Rattus norvegicus"))):
        feature = Factory.from_json(spec)
    assert feature.anchor.species == {Species.decode("Rattus norvegicus")}


def test_id_from_spec(spec):
    spec["@id"] = "long-bundles"
    assert Factory.from_json(spec).id == "long-bundles"


def test_id_depends_on_the_data_not_on_the_anchor(spec):
    spec["fragment_regions"] = {"Left_Arcuate": ["hOc1 left", "hOc2 left"]}
    first = Factory.from_json(spec).id
    spec["fragment_regions"] = {"Left_Arcuate": ["hOc2 left", "hOc1 left"]}
    assert Factory.from_json(spec).id == first
    spec["providers"] = {"trx": spec["providers"]["trx"] + "?other"}
    assert Factory.from_json(spec).id != first
