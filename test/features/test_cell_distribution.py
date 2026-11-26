from unittest.mock import patch

import numpy as np
import pytest

from siibra.configuration.factory import Factory
from siibra.features.feature import CompoundFeature
from siibra.features.tabular.point_distribution import CellDistribution

from .test_point_distribution import AFFINE, make_spec, write_tables

CELL_TYPE = "siibra/feature/point_distribution/cell_distribution/v0.1"
OTHER_DATASET = "d56d6bd3-1b24-4f1e-97c3-637bf8e9231b"


@pytest.fixture
def repo(tmp_path):
    return write_tables(tmp_path)


def cell_spec(repo_folder, **overrides):
    return make_spec(repo_folder, **{"@type": CELL_TYPE, **overrides})


@pytest.fixture
def compound():
    """Compound without resolving regions or dataset descriptions."""
    with patch.object(CellDistribution, "_merge_anchors", return_value=None), \
            patch.object(CellDistribution, "description", new=""), \
            patch.object(CompoundFeature, "description", new=""):
        yield lambda features: CompoundFeature._compound(features, None)


def two_regions(repo_folder, **overrides):
    return (
        Factory.from_json(cell_spec(repo_folder, **overrides))
        + Factory.from_json(cell_spec(repo_folder, region="Anterior olfactory nucleus", **overrides))
    )


def test_cell_type_builds_cell_distributions(repo):
    features = Factory.from_json(cell_spec(repo))
    assert all(isinstance(f, CellDistribution) for f in features)
    assert features[0].modality == "cell distribution"
    assert [f.subject for f in features] == ["1", "2"]
    assert features[0].region == "Accessory olfactory bulb"
    assert features[0].name == "Cell Distribution, subject: 1"


def test_one_compound_per_subject_with_one_element_per_region(repo, compound):
    features = two_regions(repo)
    compounds = compound(features)
    assert len(compounds) == 2
    assert [c.indices for c in compounds] == [
        ["Accessory olfactory bulb", "Anterior olfactory nucleus"],
        ["Accessory olfactory bulb", "Anterior olfactory nucleus"],
    ]
    assert compounds[1].get_element("Anterior olfactory nucleus") is features[3]


def test_compound_data_has_the_region_of_each_cell(repo, compound):
    first_subject = compound(two_regions(repo, transform={"affine": AFFINE}))[0]
    data = first_subject.data
    assert first_subject.subject == "1"
    assert data["region"].tolist() == ["Accessory olfactory bulb"] * 2 + ["Anterior olfactory nucleus"] * 2
    np.testing.assert_allclose(data["x"], [-5.0, -4.0, -5.0, -4.0])  # transformed once


def test_subjects_of_different_datasets_are_not_compounded_together(repo, compound):
    other = make_spec(repo, **{"@type": CELL_TYPE}, ebrains={
        "openminds/Species": "d9875ebd-260e-4337-a637-b62fed4aa91d",
        "openminds/DatasetVersion": OTHER_DATASET,
    })
    features = Factory.from_json(cell_spec(repo)) + Factory.from_json(other)
    assert [len(c) for c in compound(features)] == [1, 1, 1, 1]


def test_modalities_are_not_compounded_together(repo, compound):
    features = Factory.from_json(cell_spec(repo)) + Factory.from_json(cell_spec(repo, modality="parvalbumin-positive cells"))
    assert len(compound(features)) == 4


def test_compound_is_not_plotted(repo, compound):
    first_subject = compound(two_regions(repo))[0]
    with pytest.raises(NotImplementedError):
        first_subject.plot()


def test_whole_brain_tables_are_anchored_to_their_space_only(repo, compound):
    spec = cell_spec(repo)
    del spec["region"]
    features = Factory.from_json(spec)
    assert [f.region for f in features] == [None, None]
    assert [len(c) for c in compound(features)] == [1, 1]  # one compound per subject
