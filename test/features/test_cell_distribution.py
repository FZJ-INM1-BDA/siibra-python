import pytest

from siibra.configuration.factory import Factory
from siibra.features.feature import Compoundable, CompoundFeature
from siibra.features.tabular.point_distribution import CellDistribution

from .test_point_distribution import make_spec, write_tables

CELL_TYPE = "siibra/feature/point_distribution/cell_distribution/v0.1"


@pytest.fixture
def repo(tmp_path):
    return write_tables(tmp_path)


def cell_spec(repo_folder, **overrides):
    return make_spec(repo_folder, **{"@type": CELL_TYPE, **overrides})


def test_one_cell_distribution_per_subject(repo):
    features = Factory.from_json(cell_spec(repo))
    assert all(isinstance(f, CellDistribution) for f in features)
    assert [f.subject for f in features] == ["1", "2"]
    assert features[0].modality == "cell distribution"
    assert features[0].name == "Cell Distribution in Accessory olfactory bulb, subject: 1"


def test_whole_brain_tables_need_no_region(repo):
    spec = cell_spec(repo)
    del spec["region"]
    features = Factory.from_json(spec)
    assert [f.subject for f in features] == ["1", "2"]
    assert features[0].name == "Cell Distribution, subject: 1"


@pytest.mark.parametrize("subject", [None, ""])
def test_a_subject_is_required(subject):
    with pytest.raises(ValueError, match="needs a subject"):
        CellDistribution(subject=subject, anchor=None, loader=object())


def test_cell_distributions_are_not_compounded(repo):
    features = Factory.from_json(cell_spec(repo)) + Factory.from_json(cell_spec(repo, region="Anterior olfactory nucleus"))
    assert not issubclass(CellDistribution, Compoundable)
    assert CompoundFeature._compound(features, None) == features
