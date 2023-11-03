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
    "lq0::EbrainsDataFeature::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS) left::https://nexus.humanbrainproject.org/v0/data/minds/core/dataset/v1.0.0/3ff328fa-f48f-474b-bd81-b5ee7ca230b6",
    "cf0::BigBrainIntensityProfile::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS) left::nodsid::6ba1f5c180a63705d84c25db9ff7efa7",  # CompoundFeature of 1579 BigBrainIntensityProfile features grouped by (Modified silver staining modality) anchored at Area hOc1 (V1, 17, CalcS) left with Set of 1579 points in the Bounding box from (-63.69,-59.94,-29.09) mm to (0.91,77.90,54.03)mm in BigBrain microscopic template (histology) space
    "lq0::BigBrainIntensityProfile::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS) left::a48879314d93d15c74e6512e9af9c91d",  # BigBrainIntensityProfile (Modified silver staining) anchored at Area hOc1 (V1, 17, CalcS) left with Point in BigBrain microscopic template (histology) [0.4248340129852295,50.589298248291016,-14.839699745178223]
    "cf0::CellDensityProfile::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS) left::f2cd6b97-e42d-41af-a01b-5caf2d692e28::43d02182a9133fb030e4071eea539990",  # CompoundFeature of 10 CellDensityProfile features grouped by (Segmented cell body density modality) anchored at Area hOc1 (V1, 17, CalcS) with Set of 10 points in the Bounding box from (-3.95,-65.80,-0.44) mm to (20.20,-42.70,9.71)mm in BigBrain microscopic template (histology) space
    "f2cd6b97-e42d-41af-a01b-5caf2d692e28--ccc56085205beadcd4e911049e726c43",  # CellDensityProfile (Segmented cell body density) anchored at Area hOc1 (V1, 17, CalcS) with Point in BigBrain microscopic template (histology) [20.199174880981445,-64.5999984741211,-0.44111010432243347]
    "cf0::ReceptorDensityProfile::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS) left::e715e1f7-2079-45c4-a67f-f76b102acfce::1291b163f73216f756ea258f6ab2efb1",  # CompoundFeature of 16 ReceptorDensityProfile features grouped by (Receptor density modality) anchored at Area hOc1 (V1, 17, CalcS)
    "e715e1f7-2079-45c4-a67f-f76b102acfce--02200d55e4d91084e3d0014bfb9052f4",  # ReceptorDensityProfile (Receptor density) anchored at Area hOc1 (V1, 17, CalcS) for alpha4beta2
    "cf0::StreamlineCounts::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::f16e449d-86e1-408b-9487-aa9d72e39901::f09a0f43742ce628203a81029644a2c0",  # CompoundFeature of 200 StreamlineCounts features grouped by (StreamlineCounts modality, HCP cohort) anchored at Julich-Brain Cytoarchitectonic Atlas (v2.9)
    "f16e449d-86e1-408b-9487-aa9d72e39901--9cb3e151bc3584d1fa01467d72c6cb6a",  # StreamlineCounts (StreamlineCounts) anchored at minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290 with cohort HCP - 025
    "f16e449d-86e1-408b-9487-aa9d72e39901--7f6d0f1fd590eca5d0bf794ec5d05529--71b04e9f78e51fae3df339d5b20c3a51",  # FunctionalConnectivity (FunctionalConnectivity) anchored at minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290 with cohort HCP - 025 and paradigm Resting state (EmpCorrFC concatenated)
    "cf0::FunctionalConnectivity::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::f16e449d-86e1-408b-9487-aa9d72e39901::5e51be32022dd9f6f9034eef2268462e",  # CompoundFeature of 1000 FunctionalConnectivity features grouped by (FunctionalConnectivity modality, HCP cohort) anchored at Julich-Brain Cytoarchitectonic Atlas (v2.9)
    "b08a7dbc-7c75-4ce7-905b-690b2b1e8957--0b464eccb6e8afa4be9fc7a3c814e927",  # MRIVolumeOfInterest 'Fiber structures of a human hippocampus based on joint DMRI, 3D-PLI, and TPFM acquisitions (T2)' in space 'BigBrain microscopic template (histology)
]


@pytest.mark.parametrize("fid", ids)
def test_get_instance(fid):
    feat = siibra.features.Feature._get_instance_by_id(fid)
    assert feat.id == fid


# this tests whether or not calling a live query caused proxy feature to be
# added to subclasses. (It should not: causes memory leak and also increases
# query time linearly)
@pytest.mark.parametrize("fid", ids)
def test_subclass_count(fid):
    len_before = len(siibra.features.Feature.SUBCLASSES[siibra.features.Feature])
    _ = siibra.features.Feature._get_instance_by_id(fid)
    len_after = len(siibra.features.Feature.SUBCLASSES[siibra.features.Feature])
    assert len_before == len_after
