import pytest

import siibra
from siibra.core.region import Region
from siibra.features.feature import Feature
from siibra.features.tabular import FunctionalFingerprint, RegionalFunctionalActivation

func_activations = RegionalFunctionalActivation._get_instances()

# Harvard-oxford maps needs maint
# This data does not make sense for fiber bundle and sulci maps
BLACKLIST = {
    "Harvard-Oxford cortical and subcortical structural atlases",
    "Harvard-Oxford cortical and subcortical structural atlases",
    "Probabilistic map of the right temporal cingulate fibres (atlas of deep white matter fibre bundles) (2018)",
    "Morphologist Brain Sulci Atlas (2018)",
    "Atlas of the short fiber bundles inferred from the Human Connectome Project diffusion MRI dataset (v1.0)",
    "Maximum probability maps of superficial white matter fibre bundles (atlas of superficialwhite matter fibre bundles) (2018)",
    "Maximum probability maps of superficial white matter fibre bundles (atlas of superficial white matter fibre bundles) (2018)",
}
num_labelled_maps_in_mni152 = len(
    [
        mp
        for mp in siibra.maps
        if mp.is_labelled
        and mp.space == siibra.spaces["mni152"]
        and mp.name not in BLACKLIST
    ]
)


def test_one_func_activation_per_labelled_mni152_parcellation():
    parcellations = [t.parcellation for t in func_activations]
    assert len(func_activations) >= num_labelled_maps_in_mni152
    assert len(set(parcellations)) == len(parcellations)


@pytest.mark.parametrize(
    "func_activation", func_activations, ids=lambda t: t.parcellation.name
)
def test_func_activation_columns_resolve_to_regions(
    func_activation: RegionalFunctionalActivation,
):
    data = func_activation.data
    assert len(data.index) == 300
    # assert data.index.is_unique
    assert all(isinstance(c, Region) for c in data.columns)
    assert func_activation.activation_maps.index.equals(data.index)


def test_func_activation_ids_are_unique():
    ids = [t.id for t in func_activations]
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("version", ["julich 3.1", "julich 3.0.3", "julich 2.9"])
def test_region_receives_only_its_own_versions_fingerprint(version):
    region = siibra.get_region(version, "Area 4a (PreCG) left")
    fingerprints = siibra.features.get(region, FunctionalFingerprint)
    assert len(fingerprints) == 1
    assert fingerprints[0].anchor.regions
    assert all(
        r.parcellation == region.parcellation for r in fingerprints[0].anchor.regions
    )


def test_region_receives_only_its_own_parcellations_func_activation():
    region = siibra.get_region("julich 3.1", "Area 4a (PreCG) left")
    found = siibra.features.get(region, RegionalFunctionalActivation)
    assert [t.parcellation for t in found] == [region.parcellation]


def test_parent_region_receives_one_fingerprint_per_leaf():
    parent = siibra.get_region("julich 3.1", "Area 4a (PreCG)")
    fingerprints = siibra.features.get(parent, FunctionalFingerprint)
    assert {r for f in fingerprints for r in f.anchor.regions} <= set(parent.leaves)
    assert len(fingerprints) == len(parent.leaves)


def test_parcellation_receives_its_func_activation_only():
    parcellation = siibra.parcellations.get("julich 3.1")
    found = siibra.features.get(parcellation, RegionalFunctionalActivation)
    assert len(found) == 1
    assert found[0].parcellation == parcellation


def test_fingerprint_can_be_retrieved_by_id():
    region = siibra.get_region("julich 3.1", "Area 4a (PreCG) left")
    fingerprint = siibra.features.get(region, FunctionalFingerprint)[0]
    retrieved = Feature._get_instance_by_id(fingerprint.id)
    assert retrieved.id == fingerprint.id
    assert retrieved.data.equals(fingerprint.data)
