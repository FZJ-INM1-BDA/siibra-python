import copy
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from siibra.configuration.factory import Factory
from siibra.core.region import Region
from siibra.core.space import Space
from siibra.features import anchor as _anchor
from siibra.features.anchor import AnatomicalAnchor
from siibra.features.tabular.point_distribution import PointDistribution
from siibra.retrieval import repositories

MOUSE_SPACE = "minds/core/referencespace/v1.0.0/265d32a0-3d84-40a5-926f-bf89f68212b9"
DATASET = "c25f0a96-cdf0-4277-ba3c-495de13f3ccd"
TSV_DECODER = {
    "@type": "siibra/decoder/csv",
    "delimiter": "\t",
    "engine": "python",
    "names": ["x", "y", "z"],
    "index_col": False,
    "skiprows": 1,
}
AFFINE = [
    [0.025, 0.0, 0.0, -5.0],
    [0.0, 0.025, 0.0, -6.0],
    [0.0, 0.0, 0.025, -4.0],
    [0.0, 0.0, 0.0, 1.0],
]


def write_tables(folder):
    """A local repository with two tables of points."""
    (folder / "sub-1_desc-aob.tsv").write_text("x\ty\tz\n0\t0\t0\n40\t80\t120\n")
    (folder / "sub-2_desc-aob.tsv").write_text("x\ty\tz\n4\t8\t12\n")
    return folder


@pytest.fixture
def repo(tmp_path):
    return write_tables(tmp_path)


def make_spec(repo_folder, **overrides):
    spec = {
        "region": "Accessory olfactory bulb",
        "ebrains": {
            "openminds/Species": "d9875ebd-260e-4337-a637-b62fed4aa91d",
            "openminds/DatasetVersion": DATASET,
        },
        "decoder": TSV_DECODER,
        "repository": {"@type": "siibra/repository/localfolder/v1.0.0", "folder": str(repo_folder)},
        "files": {"1": "sub-1_desc-aob.tsv", "2": "sub-2_desc-aob.tsv"},
        "space": {"@id": MOUSE_SPACE},
    }
    spec.update(overrides)
    return spec


def build(spec, **kwargs):
    return Factory._build_point_distributions(spec, PointDistribution, **kwargs)


def test_one_feature_per_file(repo):
    features = build(make_spec(repo))
    assert [len(f.data) for f in features] == [2, 1]
    assert features[0].anchor is features[1].anchor


def test_key_attribute_gets_the_key_of_each_file(repo):
    built = Factory._build_point_distributions(make_spec(repo), lambda **kwargs: kwargs, key_attribute="subject")
    assert [kwargs["subject"] for kwargs in built] == ["1", "2"]
    assert "subject" not in Factory._build_point_distributions(make_spec(repo), lambda **kwargs: kwargs)[0]


def test_nothing_is_read_before_the_data_is_accessed(repo):
    spec = make_spec(repo, repository={
        "@type": "siibra/repository/dataproxy/v1.0.0",
        "bucketname": "ebrains2-t13-rodent-cell-distribution-an",
        "folder": "d56d6bd3-1b24-4f1e-97c3-637bf8e9231b",
    })
    requested, read = [], []

    class FakeRequest:
        def __init__(self, url, func=None):
            self.url = url
            requested.append(url)

        def get(self):
            read.append(self.url)
            return pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]})

    with patch.object(repositories, "BucketApiClient") as client, \
            patch.object(repositories, "HttpRequest", FakeRequest):
        features = build(spec)
        assert read == []
        assert len(features[0].data) == 1
    client.assert_not_called()  # nothing is listed
    assert requested == [
        "https://data-proxy.ebrains.eu/api/v1/buckets/ebrains2-t13-rodent-cell-distribution-an/"
        "d56d6bd3-1b24-4f1e-97c3-637bf8e9231b/" + name
        for name in ["sub-1_desc-aob.tsv", "sub-2_desc-aob.tsv"]
    ]
    assert read == requested[:1]


def test_affine_is_applied_when_reading(repo):
    feature = build(make_spec(repo, transform={"affine": AFFINE}))[0]
    np.testing.assert_allclose(
        feature.data[["x", "y", "z"]].to_numpy(),
        [[-5.0, -6.0, -4.0], [-4.0, -4.0, -1.0]],
    )
    # neither the file nor the cached table is changed
    assert (repo / "sub-1_desc-aob.tsv").read_text() == "x\ty\tz\n0\t0\t0\n40\t80\t120\n"
    np.testing.assert_allclose(feature.data["x"], [-5.0, -4.0])


def test_without_affine_coordinates_are_used_as_stored(repo):
    feature = build(make_spec(repo))[0]
    assert feature.affine is None
    assert feature.data[["x", "y", "z"]].values.tolist() == [[0, 0, 0], [40, 80, 120]]


def test_data_is_a_copy(repo):
    feature = build(make_spec(repo, transform={"affine": AFFINE}))[0]
    feature.data.loc[:, "x"] = 0
    np.testing.assert_allclose(feature.data["x"], [-5.0, -4.0])


def test_the_anchor_has_the_space_of_the_points(repo):
    feature = build(make_spec(repo))[0]
    mouse = object()
    with patch.object(_anchor, "get_registry") as get_registry:
        get_registry.return_value.get.return_value = mouse
        assert feature.space is mouse
        assert feature.anchor.space is mouse
    get_registry.return_value.get.assert_called_once_with(MOUSE_SPACE)


def test_space_queries_find_every_point_distribution_in_their_space(repo):
    mouse = MagicMock(spec=Space)
    mouse.matches.side_effect = lambda other: other is mouse
    features = build(make_spec(repo)) + build(make_spec(repo, region="Anterior olfactory nucleus"))
    with patch.object(_anchor, "get_registry") as get_registry:
        get_registry.return_value.get.return_value = mouse
        assert all(f.matches(mouse) for f in features)
        assert not any(f.matches(MagicMock(spec=Space)) for f in features)


def test_a_space_is_anchor_enough(repo):
    spec = make_spec(repo)
    del spec["region"]
    feature = build(spec)[0]
    assert isinstance(feature.anchor, AnatomicalAnchor)
    assert feature.anchor._regionspec is None and feature.anchor.location is None
    assert not feature.matches(Region("Accessory olfactory bulb"))


@pytest.mark.parametrize("change", ["base_url", "no repository", "no space", "unknown repository"])
def test_invalid_specs(repo, change):
    spec = make_spec(repo)
    if change == "base_url":
        spec["base_url"] = "https://example.org"
    elif change == "no repository":
        del spec["repository"]
    elif change == "no space":
        del spec["space"]
    else:
        spec["repository"] = {"@type": "siibra/repository/unknown/v1.0.0"}
    with pytest.raises(ValueError):
        build(spec)


def test_spec_is_not_mutated(repo):
    spec = make_spec(repo, transform={"affine": AFFINE})
    original = copy.deepcopy(spec)
    build(spec)
    assert spec == original


def test_modality_prerelease_and_name_come_from_spec(repo):
    feature = build(make_spec(repo, modality="parvalbumin-positive cells", prerelease=True))[0]
    assert feature.modality == "parvalbumin-positive cells"
    assert feature.name == "[PRERELEASE] Point Distribution"
    assert build(make_spec(repo))[0].modality == "point distribution"


def test_ids(repo):
    ids = [f.id for f in build(make_spec(repo))]
    assert len(set(ids)) == 2
    assert all(i.startswith(DATASET + "--") for i in ids)
    assert ids == [f.id for f in build(make_spec(repo, prerelease=True))]
    assert [f.id for f in build(make_spec(repo, **{"@id": "abc"}))] == ["abc--1", "abc--2"]


def test_affine_must_be_4x4(repo):
    with pytest.raises(ValueError, match="4x4"):
        build(make_spec(repo, transform={"affine": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}))


def test_plotting_is_not_implemented(repo):
    with pytest.raises(NotImplementedError):
        build(make_spec(repo))[0].plot()
