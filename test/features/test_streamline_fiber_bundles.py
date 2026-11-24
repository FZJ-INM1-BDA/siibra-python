"""Unit tests for StreamlineFiberBundles, built from specs with a local TRX file."""

import json
import logging
from unittest.mock import Mock, patch

import numpy as np
import pytest

pytest.importorskip("trx", reason="trx-python requires Python >= 3.11")

from siibra.commons import Species  # noqa: E402
from siibra.configuration.factory import Factory  # noqa: E402
from siibra.core import space  # noqa: E402
from siibra.core.region import Region  # noqa: E402
from siibra.features.anchor import AnatomicalAnchor  # noqa: E402
from siibra.features.connectivity import StreamlineFiberBundles  # noqa: E402
from siibra.retrieval import requests  # noqa: E402
from test.features.streamline_files import STREAMLINES, write_trx  # noqa: E402


@pytest.fixture
def spec(tmp_path):
    return {
        "@type": "siibra/feature/streamlinefiberbundles/v0.2",
        "name": "Long fiber bundles",
        "modality": "Streamline long fiber bundles",
        "species": "Homo sapiens",
        "space": {"@id": "test-space"},
        "trx": write_trx(tmp_path / "bundles.trx"),
    }


@pytest.fixture
def feature(spec):
    return Factory.from_json(spec)


def streamline_indices(trx):
    """Indices into STREAMLINES of the streamlines in a TRX file."""
    found = []
    for streamline in trx.streamlines:
        found.append(next(i for i, s in enumerate(STREAMLINES) if s.shape == streamline.shape and np.allclose(s, streamline)))
    return found


def test_spec_matches_the_schema(spec, monkeypatch):
    jsonschema = pytest.importorskip("jsonschema")
    monkeypatch.syspath_prepend("config_schema")
    from check_schema import get_ref
    with open("config_schema/siibra/feature/streamlinefiberbundles/v0.2.json") as f:
        schema = json.load(f)
    jsonschema.validate(spec, schema, resolver=get_ref(schema))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({k: v for k, v in spec.items() if k != "trx"}, schema, resolver=get_ref(schema))


def test_builds_from_spec(feature, spec):
    assert isinstance(feature, StreamlineFiberBundles)
    assert feature.name == "Long fiber bundles (Streamline long fiber bundles)"
    assert feature.modality == "Streamline long fiber bundles"
    assert feature._loader.url == spec["trx"]


def test_anchored_to_species_only(feature):
    assert feature.anchor.species == {Species.decode("Homo sapiens")}
    assert feature.anchor.regions == {}
    assert feature.anchor.location is None
    assert not feature.matches(Region("hOc1 left"))


@pytest.mark.parametrize("species_spec, expected", [
    ({"ebrains": {"openminds/Species": "97c070c6-8e1f-4ee8-9d28-18c7945921dd"}}, "Homo sapiens"),
    ({}, "Rattus norvegicus"),  # from the space
])
def test_species_sources(spec, species_spec, expected):
    del spec["species"]
    spec.update(species_spec)
    with patch.object(space.Space, "get_instance", return_value=Mock(species=Species.decode("Rattus norvegicus"))):
        feature = Factory.from_json(spec)
    assert feature.anchor.species == {Species.decode(expected)}


def test_bundles_are_the_sorted_trx_groups(feature):
    assert feature.bundles == ["Left_Arcuate", "Left_Arcuate_anterior", "Right_Uncinate"]


def test_fetch_all_bundles_returns_the_loaded_file(feature):
    trx = feature.fetch()
    assert trx is feature.fetch()
    assert len(trx.streamlines) == len(STREAMLINES)


def test_fetch_bundles_keeps_only_requested_groups(feature):
    trx = feature.fetch(["Left_Arcuate"])
    assert streamline_indices(trx) == [0, 1]
    # "Left_Arcuate_anterior" shares streamline 1, but is not requested
    assert list(trx.groups) == ["Left_Arcuate"]


def test_fetch_overlapping_bundles_returns_each_streamline_once(feature):
    trx = feature.fetch(["Left_Arcuate", "Left_Arcuate_anterior"])
    assert streamline_indices(trx) == [0, 1, 2]
    assert sorted(trx.groups) == ["Left_Arcuate", "Left_Arcuate_anterior"]


def test_fetch_accepts_a_single_name(feature):
    assert streamline_indices(feature.fetch("Right_Uncinate")) == [3, 4]


@pytest.mark.parametrize("request_", ["left_arcuate", "Arcuate", ["Right_Uncinate", "Corpus_Callosum"]])
def test_fetch_matches_names_exactly(feature, request_):
    with pytest.raises(ValueError, match="has no bundle"):
        feature.fetch(request_)


def test_selection_does_not_log_trx_group_warnings(feature, caplog):
    with caplog.at_level(logging.WARNING):
        feature.fetch(["Left_Arcuate"])
    assert "dpg" not in caplog.text


def test_trx_is_loaded_once_per_loader(feature):
    get = Mock(wraps=feature._loader.get)
    with patch.object(feature._loader, "get", get):
        StreamlineFiberBundles._TRX_CACHE.pop(feature._loader, None)
        feature.fetch()
        feature.fetch(["Right_Uncinate"])
        feature._filter_by_bundles(["Right_Uncinate"], feature.anchor).fetch()
    assert get.call_count == 1


def test_filtered_feature_knows_its_bundles_without_loading(feature):
    filtered = feature._filter_by_bundles(["Right_Uncinate", "Left_Arcuate"], AnatomicalAnchor(species="Homo sapiens"))
    with patch.object(StreamlineFiberBundles, "_load", side_effect=AssertionError("loaded")):
        assert filtered.bundles == ["Right_Uncinate", "Left_Arcuate"]
    assert filtered.name == feature.name
    assert filtered._loader is feature._loader


def test_filtered_feature_fetches_only_its_bundles(feature):
    filtered = feature._filter_by_bundles(["Left_Arcuate", "Right_Uncinate"], feature.anchor)
    assert streamline_indices(filtered.fetch()) == [0, 1, 3, 4]
    with pytest.raises(ValueError, match="has no bundle"):
        filtered.fetch(["Left_Arcuate_anterior"])


def test_bundles_missing_from_the_trx_are_skipped_with_a_warning(feature, caplog):
    filtered = feature._filter_by_bundles(["Left_Arcuate", "Corpus_Callosum"], feature.anchor)
    with caplog.at_level(logging.WARNING, logger="siibra"):
        assert streamline_indices(filtered.fetch()) == [0, 1]
    assert "Corpus_Callosum" in caplog.text


def test_missing_trx_python_is_explained(feature, monkeypatch):
    monkeypatch.setattr(requests, "_HAS_TRX", False)
    with pytest.raises(ImportError, match="Python >= 3.11"):
        feature.fetch()
