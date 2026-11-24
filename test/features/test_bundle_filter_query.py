"""Unit tests for BundleFilterQuery, with a local folder standing in for the data-proxy bucket."""

import logging
from unittest.mock import Mock, patch

import pandas as pd
import pytest

pytest.importorskip("trx", reason="trx-python requires Python >= 3.11")

import siibra  # noqa: E402
from siibra.configuration.factory import Factory  # noqa: E402
from siibra.core import space  # noqa: E402
from siibra.core.assignment import Qualification  # noqa: E402
from siibra.core.parcellation import Parcellation  # noqa: E402
from siibra.core.region import Region  # noqa: E402
from siibra.features.connectivity import StreamlineFiberBundles  # noqa: E402
from siibra.livequeries import BundleFilterQuery  # noqa: E402
from siibra.retrieval.repositories import LocalFileRepository  # noqa: E402
from siibra.volumes.volume import Volume  # noqa: E402
from test.features.streamline_files import write_trx  # noqa: E402

HOC1_LEFT, HOC1_RIGHT, HOC2_LEFT, AREA44_LEFT = (
    Region("hOc1 left"), Region("hOc1 right"), Region("hOc2 left"), Region("Area 44 left")
)
HOC1 = Region("hOc1", children=[HOC1_LEFT, HOC1_RIGHT])
PARCELLATION = Parcellation(
    identifier="bundle-filter-test-parcellation", name="bundle filter test parcellation",
    species="Homo sapiens", regions=[HOC1, HOC2_LEFT, AREA44_LEFT],
)
OTHER_PARCELLATION = Region("other test parcellation", children=[Region("hOc1 left")])

TABLE = pd.DataFrame(
    [["Left_Arcuate", 1, 0, 0, 0],
     ["Left_Arcuate_anterior", 0, 1, 1, 0],
     ["Right_Uncinate", 0, 0, 1, 0]],
    columns=["bundle", "hOc1 left", "hOc1 right", "hOc2 left", "Area 44 left"],
)
MODALITY = "Streamline long fiber bundles"


@pytest.fixture
def bucket(tmp_path):
    folder = tmp_path / "bucket" / "terminal_region_tables" / "streamline_long_fiber_bundles"
    folder.mkdir(parents=True)
    TABLE.to_csv(folder / "MNI152_TEST_LABELLED_connectivity_matrix_long_bundles.tsv", sep="\t", index=False)
    # another parcellation's table, whose key contains this one's
    TABLE.iloc[:, :2].to_csv(folder / "MNI152_TEST2_LABELLED_connectivity_matrix_long_bundles.tsv", sep="\t", index=False)
    return LocalFileRepository(str(tmp_path / "bucket"))


@pytest.fixture
def tractogram(tmp_path):
    return Factory.from_json({
        "@type": "siibra/feature/streamlinefiberbundles/v0.2",
        "name": "Long fiber bundles",
        "modality": MODALITY,
        "species": "Homo sapiens",
        "space": {"@id": "test-space"},
        "trx": write_trx(tmp_path / "bundles.trx"),
    })


@pytest.fixture(autouse=True)
def setup(bucket, tractogram, monkeypatch):
    monkeypatch.setattr(BundleFilterQuery, "_connector", classmethod(lambda cls: bucket))
    monkeypatch.setattr(BundleFilterQuery, "_tables", {})
    monkeypatch.setattr(BundleFilterQuery, "PARCELLATION_KEYS", {PARCELLATION.id: "TEST"})
    monkeypatch.setattr(StreamlineFiberBundles, "_get_instances", classmethod(lambda cls, **kw: [tractogram]))


def query(concept):
    return BundleFilterQuery().query(concept)


def test_modality_folder():
    assert BundleFilterQuery.modality_folder("Streamline long fiber bundles") == "streamline_long_fiber_bundles"
    assert BundleFilterQuery.modality_folder("U-fibers (short)") == "u_fibers_short"


def test_leaf_region_yields_the_bundles_terminating_in_it(tractogram):
    [feature] = query(HOC1_LEFT)
    assert feature.bundles == ["Left_Arcuate"]
    assert feature._loader is tractogram._loader
    assert len(feature.fetch().streamlines) == 2


def test_anchor_is_the_query_region_qualified_as_overlapping():
    [feature] = query(HOC2_LEFT)
    assert feature.anchor.regions == {HOC2_LEFT: Qualification.OVERLAPS}
    assert feature.matches(HOC2_LEFT)


def test_non_leaf_region_yields_bundles_terminating_in_its_leaves():
    [feature] = query(HOC1)
    assert feature.bundles == ["Left_Arcuate", "Left_Arcuate_anterior"]
    assert feature.anchor.regions == {HOC1: Qualification.OVERLAPS}


def test_parcellation_yields_bundles_terminating_anywhere():
    [feature] = query(PARCELLATION)
    assert feature.bundles == ["Left_Arcuate", "Left_Arcuate_anterior", "Right_Uncinate"]


def test_region_without_terminations_yields_nothing():
    assert query(AREA44_LEFT) == []


def test_region_names_are_matched_exactly(monkeypatch):
    left = Region("hOc1 Left")
    parcellation = Region("case test parcellation", children=[left])
    monkeypatch.setattr(BundleFilterQuery, "PARCELLATION_KEYS", {parcellation.id: "TEST"})
    assert query(left) == []


def test_parcellation_without_tables_yields_nothing():
    assert query(OTHER_PARCELLATION.children[0]) == []


def test_modality_without_tables_yields_nothing(tractogram, monkeypatch):
    other = tractogram._filter_by_bundles(None, tractogram.anchor)
    other._modality_cached = "Streamline short fiber bundles"
    monkeypatch.setattr(StreamlineFiberBundles, "_get_instances", classmethod(lambda cls, **kw: [other]))
    assert query(HOC1_LEFT) == []


def test_tables_are_read_once(bucket):
    with patch.object(bucket, "get_loader", wraps=bucket.get_loader) as get_loader:
        query(HOC1_LEFT)
        query(HOC2_LEFT)
    assert get_loader.call_count == 1


def test_unreachable_bucket_yields_nothing(monkeypatch, caplog):
    def fail(cls):
        raise ConnectionError("bucket down")

    monkeypatch.setattr(BundleFilterQuery, "_connector", classmethod(fail))
    with caplog.at_level(logging.WARNING, logger="siibra"):
        assert query(HOC1_LEFT) == []
    assert "bucket down" in caplog.text


def test_space_query_yields_whole_tractograms(tractogram):
    template = Mock(spec=Volume, space=Mock(id="test-space"))
    other_template = Mock(spec=Volume, space=Mock(id="other-space"))
    with patch.object(space.Space, "get_instance", return_value=Mock(id="test-space")):
        assert query(template) == [tractogram]
        assert query(other_template) == []


def test_other_concepts_yield_nothing():
    assert query(siibra.Point((0, 0, 0), space=None)) == []


def test_features_get_returns_filtered_features_with_live_query_ids():
    [feature] = siibra.features.get(HOC1_LEFT, StreamlineFiberBundles)
    assert feature.bundles == ["Left_Arcuate"]
    assert feature.id.startswith("lq0::")
