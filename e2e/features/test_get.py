import pytest
import siibra


# We get all registered subclasses of Feature
@pytest.mark.parametrize(
    "Cls", [Cls for Cls in siibra.features.Feature.SUBCLASSES[siibra.features.Feature]]
)
def test_get_instances(Cls: siibra.features.Feature):
    instances = Cls.get_instances()
    assert isinstance(instances, list)


ids = [
    (
        "lq0::EbrainsDataFeature::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS) left::https://nexus.humanbrainproject.org/v0/data/minds/core/dataset/v1.0.0/3ff328fa-f48f-474b-bd81-b5ee7ca230b6",
        None,
    ),
    (
        "cf0::BigBrainIntensityProfile::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS) left::nodsid::ff4271d32d8b6dd556e1ebaa91f09045",
        None
    ),  # CompoundFeature of 1579 BigBrainIntensityProfile features grouped by (Modified silver staining) anchored at Area hOc1 (V1, 17, CalcS) left with Set of 1579 points in the Bounding box from (-63.69,-59.94,-29.09) mm to (0.91,77.90,54.03)mm in BigBrain microscopic template (histology) space
    (
        "lq0::BigBrainIntensityProfile::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS) left::303197094c5227c245bec8ff34191522",
        None
    ),  # BigBrainIntensityProfile queried with Area hOc1 (V1, 17, CalcS) left JBA3 and anchored at Point in BigBrain microscopic template (histology) [-1.587149977684021,69.70700073242188,6.023950099945068]
    (
        "b08a7dbc-7c75-4ce7-905b-690b2b1e8957--0b464eccb6e8afa4be9fc7a3c814e927",
        None
    ),  # MRIVolumeOfInterest 'Fiber structures of a human hippocampus based on joint DMRI, 3D-PLI, and TPFM acquisitions (T2)' in space 'BigBrain microscopic template (histology)
    (
        "cf0::CellDensityProfile::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS) left::dc358cb8-2bbb-40f1-998c-356c9e13e4c6::cbc9f7824a81db1ba00deb53c84ec3f7",
        None
    ),  # CompoundFeature of 10 CellDensityProfile features grouped by (Segmented cell body density) anchored at Area hOc1 (V1, 17, CalcS) with Set of 10 points in the Bounding box from (-3.95,-65.80,-0.44) mm to (20.20,-42.70,9.71)mm in BigBrain microscopic template (histology) space
    (
        "dc358cb8-2bbb-40f1-998c-356c9e13e4c6--45a18a7f9c7610b65148136046689234",
        None
    ),  # CellDensityProfile (Segmented cell body density) anchored at Area hOc1 (V1, 17, CalcS) with Point in BigBrain microscopic template (histology) [13.53404426574707,-64.30000305175781,5.984400749206543]
    (
        "cf0::ReceptorDensityProfile::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS) left::e715e1f7-2079-45c4-a67f-f76b102acfce::a264817171736834d75fffec45ba1757",
        None
    ),  # CompoundFeature of 16 ReceptorDensityProfile features grouped by (Receptor density) anchored at Area hOc1 (V1, 17, CalcS)
    (
        "e715e1f7-2079-45c4-a67f-f76b102acfce--402ff9f8032f5b39bdbd1a9a1c4fe1c0",
        None
    ),  # ReceptorDensityProfile (AMPA (alpha-amino-3hydroxy-5-methyl-4isoxazolepropionic acid receptor) density) anchored at Area hOc1 (V1, 17, CalcS) for
    (
        "cf0::StreamlineCounts::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-300::0f1ccc4a-9a11-4697-b43f-9c9c8ac543e6::6c085751ff0a92d47f967428720e1fe9",
        None
    ),  # CompoundFeature of 200 StreamlineCounts features grouped by (StreamlineCounts, HCP) anchored at Julich-Brain Cytoarchitectonic Atlas (v3.0.3)
    (
        "0f1ccc4a-9a11-4697-b43f-9c9c8ac543e6--5c040dd84fe23933624732e264d3d137",
        None
    )  # StreamlineCounts (StreamlineCounts) anchored at minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-300 with cohort HCP - 005
]


@pytest.mark.parametrize("fid,foo", ids)
def test_get_instance(fid, foo):
    feat = siibra.features.Feature._get_instance_by_id(fid)
    assert feat


# this tests whether or not calling a live query caused proxy feature to be
# added to subclasses. (It should not: causes memory leak and also increases
# query time linearly)
@pytest.mark.parametrize("fid,foo", ids)
def test_subclass_count(fid, foo):
    len_before = len(siibra.features.Feature.SUBCLASSES[siibra.features.Feature])
    _ = siibra.features.Feature._get_instance_by_id(fid)
    len_after = len(siibra.features.Feature.SUBCLASSES[siibra.features.Feature])
    assert len_before == len_after
