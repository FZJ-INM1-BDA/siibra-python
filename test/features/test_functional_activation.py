from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

import siibra
from siibra.features.anchor import AnatomicalAnchor
from siibra.features.tabular import FunctionalFingerprint, RegionalFunctionalActivation
from siibra.livequeries.functional_fingerprint import FunctionalFingerprintQuery

PARENT = "Area 4a (PreCG)"
MAPS = ["map a", "map b", "map c"]


def make_func_activation(
    parcellation_spec: str, offset: float, empty_leaf: bool = False,
) -> RegionalFunctionalActivation:
    """A func_activation for the leaves of area 4a, with values unique to the parcellation."""
    parcellation = siibra.parcellations.get(parcellation_spec)
    leaves = [r.name for r in parcellation.get_region(PARENT).leaves]
    data = pd.DataFrame(
        {name: [offset + i for i in range(len(MAPS))] for name in leaves},
        index=pd.Index(MAPS, name="activation_map"),
    )
    if empty_leaf:
        data[leaves[-1]] = np.nan
    func_activation = RegionalFunctionalActivation(
        file="https://example.org/func_activation.tsv",
        decoder=None,
        anchor=AnatomicalAnchor(region=parcellation, species="Homo sapiens"),
        description="test regional functional activation",
    )
    func_activation._loader = Mock(get=lambda: data.copy())
    return func_activation


def leaf_of(func_activation: RegionalFunctionalActivation, i: int):
    """The region object of the i-th column."""
    return func_activation.parcellation.get_region(func_activation.data.columns[i])


@pytest.fixture(scope="module")
def func_activations():
    return {
        "julich 3.1": make_func_activation("julich 3.1", offset=31.0),
        "julich 3.0.3": make_func_activation("julich 3.0.3", offset=303.0),
    }


@pytest.fixture
def registered(func_activations):
    with patch.object(
        RegionalFunctionalActivation, "_get_instances", return_value=list(func_activations.values())
    ):
        yield func_activations


def test_data_holds_the_region_columns(func_activations):
    func_activation = func_activations["julich 3.1"]
    leaves = func_activation.parcellation.get_region(PARENT).leaves
    assert list(func_activation.data.columns) == [r.name for r in leaves]
    assert list(func_activation.data.index) == MAPS
    assert func_activation.data.index.name == "activation_map"


def test_func_activation_matches_its_parcellation_and_regions(func_activations):
    func_activation = func_activations["julich 3.1"]
    parcellation = siibra.parcellations.get("julich 3.1")
    assert func_activation.matches(parcellation)
    assert func_activation.matches(parcellation.get_region(PARENT))
    assert func_activation.matches(leaf_of(func_activation, 0))


def test_func_activation_does_not_match_other_parcellation_version(func_activations):
    func_activation = func_activations["julich 3.1"]
    other = siibra.parcellations.get("julich 3.0.3")
    assert not func_activation.matches(other)
    assert not func_activation.matches(other.get_region(PARENT))


def test_func_activation_name_includes_parcellation(func_activations):
    func_activation = func_activations["julich 3.1"]
    assert func_activation.parcellation.name in func_activation.name


@pytest.mark.parametrize("version", ["julich 3.1", "julich 3.0.3"])
def test_fingerprint_comes_from_own_parcellation_version(registered, version):
    func_activation = registered[version]
    leaf = leaf_of(func_activation, 0)
    fingerprints = FunctionalFingerprintQuery().query(leaf)
    assert len(fingerprints) == 1
    fingerprint = fingerprints[0]
    assert fingerprint.region is leaf
    assert fingerprint.region.parcellation == func_activation.parcellation
    assert fingerprint.data["mean"].tolist() == func_activation.data[leaf.name].tolist()


def test_parent_region_receives_one_fingerprint_per_leaf(registered):
    parent = siibra.parcellations.get("julich 3.1").get_region(PARENT)
    fingerprints = FunctionalFingerprintQuery().query(parent)
    assert {f.region for f in fingerprints} == set(parent.leaves)
    assert all(f.region.is_leaf for f in fingerprints)
    assert all(list(f.data.columns) == ["mean"] for f in fingerprints)


def test_leaf_without_data_receives_no_fingerprint():
    func_activation = make_func_activation("julich 3.1", offset=0.0, empty_leaf=True)
    parent = func_activation.parcellation.get_region(PARENT)
    with patch.object(RegionalFunctionalActivation, "_get_instances", return_value=[func_activation]):
        fingerprints = FunctionalFingerprintQuery().query(parent)
    empty = leaf_of(func_activation, -1)
    assert empty not in {f.region for f in fingerprints}
    assert len(fingerprints) == len(parent.leaves) - 1


def test_no_fingerprints_for_parcellations_spaces_or_unrelated_regions(registered):
    query = FunctionalFingerprintQuery()
    assert query.query(siibra.parcellations.get("julich 3.1")) == []
    assert query.query(siibra.spaces.get("mni152")) == []
    unrelated = siibra.get_region("julich 3.1", "hoc1 left")
    assert query.query(unrelated) == []


def test_fingerprint_data_and_metadata(registered):
    func_activation = registered["julich 3.1"]
    leaf = leaf_of(func_activation, 0)
    fingerprint = FunctionalFingerprint(region=leaf, func_activation=func_activation)
    assert list(fingerprint.data.columns) == ["mean"]
    assert list(fingerprint.data.index) == MAPS
    assert fingerprint.modality == "functional fingerprint"
    assert fingerprint.unit == "z"
    assert fingerprint.anchor.species == func_activation.anchor.species


def test_fingerprint_ids_are_unique_and_stable(registered):
    func_activation = registered["julich 3.1"]
    leaves = func_activation.parcellation.get_region(PARENT).leaves
    ids = [FunctionalFingerprint(region=r, func_activation=func_activation).id for r in leaves]
    assert len(set(ids)) == len(ids)
    assert FunctionalFingerprint(region=leaves[0], func_activation=func_activation).id == ids[0]


def test_features_get_serves_fingerprints_through_live_query(registered):
    leaf = leaf_of(registered["julich 3.1"], 0)
    features = siibra.features.get(leaf, FunctionalFingerprint)
    assert len(features) == 1
    assert features[0].id.startswith("lq0::FunctionalFingerprint::")


def test_features_get_returns_func_activation_for_a_region_of_its_parcellation(registered):
    leaf = leaf_of(registered["julich 3.1"], 0)
    found = siibra.features.get(leaf, RegionalFunctionalActivation)
    assert [t.parcellation for t in found] == [leaf.parcellation]


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_plots(registered, backend):
    func_activation = registered["julich 3.1"]
    func_activation.plot(backend=backend)
    FunctionalFingerprint(region=leaf_of(func_activation, 0), func_activation=func_activation).plot(backend=backend)
