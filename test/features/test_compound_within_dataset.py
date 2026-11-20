from dataclasses import dataclass

import pytest

from siibra.features.feature import Compoundable, CompoundFeature


@dataclass(frozen=True)
class Dataset:
    id: str


class Measurement(Compoundable):
    """A minimal compoundable, so no Feature subclass is registered globally."""

    _filter_attrs = ["modality", "subject"]
    _compound_attrs = ["modality"]
    category = None

    def __init__(self, subject, dataset_ids, modality="measurement"):
        self.subject = subject
        self.modality = modality
        self.datasets = [Dataset(i) for i in dataset_ids]
        self.anchor = None
        self.description = ""
        self._prerelease = False

    @classmethod
    def _merge_anchors(cls, anchors):
        return None

    @classmethod
    def _merge_elements(cls, elements, description, modality, anchor):
        raise NotImplementedError


class MeasurementWithinDataset(Measurement):
    _compound_within_dataset = True


@pytest.fixture
def from_two_datasets():
    def make(cls):
        return [cls("1", ["a"]), cls("2", ["a"]), cls("1", ["b"])]
    return make


def test_datasets_are_compounded_together_by_default(from_two_datasets):
    features = from_two_datasets(Measurement)[:2] + [Measurement("3", ["b"])]
    (compound,) = CompoundFeature._compound(features, None)
    assert compound.indices == ["1", "2", "3"]


def test_default_compounding_still_needs_unique_indices(from_two_datasets):
    # Subject "1" is in both datasets: one compound cannot hold both.
    with pytest.raises(AssertionError):
        CompoundFeature._compound(from_two_datasets(Measurement), None)


def test_compounding_within_dataset_splits_by_dataset(from_two_datasets):
    compounds = CompoundFeature._compound(from_two_datasets(MeasurementWithinDataset), None)
    assert [c.indices for c in compounds] == [["1", "2"], ["1"]]
    assert [[ds.id for ds in c.datasets] for c in compounds] == [["a"], ["b"]]


def test_equal_dataset_sets_are_compounded_together():
    features = [
        MeasurementWithinDataset("1", ["a", "shared"]),
        MeasurementWithinDataset("2", ["shared", "a"]),
        MeasurementWithinDataset("3", ["shared"]),
    ]
    compounds = CompoundFeature._compound(features, None)
    assert [c.indices for c in compounds] == [["1", "2"], ["3"]]


def test_compound_key_still_separates_modalities():
    features = [
        MeasurementWithinDataset("1", ["a"], modality="x"),
        MeasurementWithinDataset("1", ["a"], modality="y"),
    ]
    assert len(CompoundFeature._compound(features, None)) == 2
