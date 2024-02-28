import pytest
import siibra


# We get all registered subclasses of Feature
@pytest.mark.parametrize(
    "Cls", [Cls for Cls in siibra.features.Feature._SUBCLASSES[siibra.features.Feature]]
)
def test_get_instances(Cls: siibra.features.Feature):
    instances = Cls._get_instances()
    assert isinstance(instances, list)


selected_ids = [
    "lq0::EbrainsDataFeature::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS) left::https://nexus.humanbrainproject.org/v0/data/minds/core/dataset/v1.0.0/3ff328fa-f48f-474b-bd81-b5ee7ca230b6",
    "cf0::BigBrainIntensityProfile::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS) left::nodsid::43ade4d58c0909df996183256e45070d",  # CompoundFeature of 1579 BigBrainIntensityProfile features grouped by (Modified silver staining modality) anchored at Area hOc1 (V1, 17, CalcS) left with Set of 1579 points in the Bounding box from (-31.80,-68.85,-12.52) mm to (5.09,-29.20,12.00)mm in BigBrain microscopic template (histology) space
    "lq0::BigBrainIntensityProfile::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS) left::0ba613a8aa7eb6a888c88485b8cd715d",  # BigBrainIntensityProfile (Modified silver staining) anchored at Area hOc1 (V1, 17, CalcS) left with Point in BigBrain microscopic template (histology) [-5.872700214385986,-55.385398864746094,-1.3151400089263916]
    "cf0::CellDensityProfile::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS) left::f2cd6b97-e42d-41af-a01b-5caf2d692e28::43d02182a9133fb030e4071eea539990",  # CompoundFeature of 10 CellDensityProfile features grouped by (Segmented cell body density modality) anchored at Area hOc1 (V1, 17, CalcS) with Set of 10 points in the Bounding box from (-3.95,-65.80,-0.44) mm to (20.20,-42.70,9.71)mm in BigBrain microscopic template (histology) space
    "f2cd6b97-e42d-41af-a01b-5caf2d692e28--ccc56085205beadcd4e911049e726c43",  # CellDensityProfile (Segmented cell body density) anchored at Area hOc1 (V1, 17, CalcS) with Point in BigBrain microscopic template (histology) [20.199174880981445,-64.5999984741211,-0.44111010432243347]
    "cf0::ReceptorDensityProfile::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS) left::e715e1f7-2079-45c4-a67f-f76b102acfce::1291b163f73216f756ea258f6ab2efb1",  # CompoundFeature of 16 ReceptorDensityProfile features grouped by (Receptor density modality) anchored at Area hOc1 (V1, 17, CalcS)
    "e715e1f7-2079-45c4-a67f-f76b102acfce--02200d55e4d91084e3d0014bfb9052f4",  # ReceptorDensityProfile (Receptor density) anchored at Area hOc1 (V1, 17, CalcS) for alpha4beta2
    "cf0::StreamlineCounts::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::f16e449d-86e1-408b-9487-aa9d72e39901::f09a0f43742ce628203a81029644a2c0",  # CompoundFeature of 200 StreamlineCounts features grouped by (StreamlineCounts modality, HCP cohort) anchored at Julich-Brain Cytoarchitectonic Atlas (v2.9)
    "f16e449d-86e1-408b-9487-aa9d72e39901--283aaf98c0a6bfd2a272a6ef7ac81dd8",  # StreamlineCounts (StreamlineCounts) anchored at Julich-Brain Cytoarchitectonic Atlas (v2.9) with cohort HCP - 025
    "3f179784-194d-4795-9d8d-301b524ca00a--713b6b5ddc0136acb757863a8138f85e--9c08356ec0454773885ded630e49b5d3",  # FunctionalConnectivity (FunctionalConnectivity) anchored at Julich-Brain Cytoarchitectonic Atlas (v2.9) with cohort 1000BRAINS - 0108_1, Resting state (RestEmpCorrFC) paradigm
    "cf0::FunctionalConnectivity::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::3f179784-194d-4795-9d8d-301b524ca00a::0cc40894189c89488637f35554f88da5",  # CompoundFeature of 349 FunctionalConnectivity features grouped by (FunctionalConnectivity modality, 1000BRAINS cohort, Resting state (RestEmpCorrFC) paradigm) anchored at Julich-Brain Cytoarchitectonic Atlas (v2.9)
    "b08a7dbc-7c75-4ce7-905b-690b2b1e8957--8ff1e1d8bcb26296027b475ec744b83c",  # Fiber structures of a human hippocampus based on joint DMRI, 3D-PLI, and TPFM acquisitions (T2 weighted MRI)
]


@pytest.mark.parametrize("fid", selected_ids)
def test_get_instance(fid):
    feat = siibra.features.Feature._get_instance_by_id(fid)
    assert feat.id == fid


# this tests whether or not calling a live query caused proxy feature to be
# added to subclasses. (It should not: causes memory leak and also increases
# query time linearly)
@pytest.mark.parametrize("fid", selected_ids)
def test_subclass_count(fid):
    len_before = len(siibra.features.Feature._SUBCLASSES[siibra.features.Feature])
    _ = siibra.features.Feature._get_instance_by_id(fid)
    len_after = len(siibra.features.Feature._SUBCLASSES[siibra.features.Feature])
    assert len_before == len_after


def test_querying_with_volume():
    # Use features with location anchors only. Because hybrid ones will also
    # employ sementic links between regions, potentially changing the result.
    region = siibra.get_region("julich 2.9", "ca1")
    volume = region.get_regional_map('mni152')
    profiles_region = siibra.features.get(region, "BigBrainIntensityProfile")[0]
    profiles_volume = siibra.features.get(volume, "BigBrainIntensityProfile")[0]
    # the ids will be diffent but the content has to be the same. Even the order.
    assert [p.location for p in profiles_region] == [p.location for p in profiles_volume]
