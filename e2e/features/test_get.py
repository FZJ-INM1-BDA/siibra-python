import pytest
import siibra

# We get all registered subclasses of Feature
@pytest.mark.parametrize('Cls', [Cls for Cls in siibra.features.Feature.SUBCLASSES[siibra.features.Feature]])
def test_get_instances(Cls: siibra.features.Feature):
    instances = Cls.get_instances()
    assert isinstance(instances, list)

ids = [
    ("lq0::EbrainsDataFeature::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS) left::https://nexus.humanbrainproject.org/v0/data/minds/core/dataset/v1.0.0/3ff328fa-f48f-474b-bd81-b5ee7ca230b6", None),
    pytest.param(
        "lq0::BigBrainIntensityProfile::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS)::f4380d69a9636d01398238b9ca602d29",None,
        marks=pytest.mark.xfail(
            reason="BigBrainIntensityProfile ids are non deterministic... somehow..."
        )
    ),
    ("e715e1f7-2079-45c4-a67f-f76b102acfce--2db407e630b9eaefa014a5a7fd506207",None),
]

@pytest.mark.parametrize('fid,foo', ids)
def test_get_instance(fid,foo):
    feat = siibra.features.Feature.get_instance_by_id(fid)
    assert feat

@pytest.mark.parametrize('fid,foo', ids)
def test_subclass_count(fid,foo):
    len_before = len(siibra.features.Feature.SUBCLASSES[siibra.features.Feature])
    feat = siibra.features.Feature.get_instance_by_id(fid)
    len_after = len(siibra.features.Feature.SUBCLASSES[siibra.features.Feature])
    assert len_before == len_after
