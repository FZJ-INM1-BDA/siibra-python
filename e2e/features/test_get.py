import pytest
from e2e.util import check_duplicate

import siibra
from siibra.locations import Location, PointCloud
from siibra.livequeries.bigbrain import WagstylProfileLoader
from siibra.features.feature import CompoundFeature

import numpy as np


# We get all registered subclasses of Feature
@pytest.mark.parametrize(
    "Cls", [Cls for Cls in siibra.features.Feature._SUBCLASSES[siibra.features.Feature]]
)
def test_get_instances(Cls: siibra.features.Feature):
    instances = Cls._get_instances()
    assert isinstance(instances, list)


@pytest.mark.parametrize(
    "Cls", [Cls for Cls in siibra.features.Feature._SUBCLASSES[siibra.features.Feature]]
)
def test_id_unique(Cls: siibra.features.Feature):
    instances = Cls._get_instances()
    duplicates = check_duplicate([f.id for f in instances])
    assert len(duplicates) == 0


@pytest.mark.parametrize(
    "Cls", [Cls for Cls in siibra.features.Feature._SUBCLASSES[siibra.features.Feature]]
)
def test_feature_unique(Cls: siibra.features.Feature):
    instances = Cls._get_instances()
    duplicates = check_duplicate([f for f in instances])
    assert len(duplicates) == 0


selected_ids = [
    "lq0::EbrainsDataFeature::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS) left::https://nexus.humanbrainproject.org/v0/data/minds/core/dataset/v1.0.0/3ff328fa-f48f-474b-bd81-b5ee7ca230b6",
    "cf0::BigBrainIntensityProfile::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS) left::acc39db40e08a9ce23d05bf75a4ce172::0d023118e0b9923a11628642e6170d00",  # 1574 BigBrain Intensity Profile features
    "lq0::BigBrainIntensityProfile::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS) left::acc39db40e08a9ce23d05bf75a4ce172--4b4e7dea27709f3367dd468f9c9a96ed",  # BigBrain Intensity Profile: (4.848050117492676, -55.64030075073242, 1.999250054359436)
    "cf0::CellDensityProfile::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS)::f2cd6b97-e42d-41af-a01b-5caf2d692e28::599f219267d5bdc3c5c04ddf31f36748",  # 10 Cell Density Profile features
    "f2cd6b97-e42d-41af-a01b-5caf2d692e28--5fc6ebfcbdf43c1c9fb36263eda160d2",  # Cell Density Profile: (19.09100914001465, -64.9000015258789, -0.36307409405708313)
    "cf0::ReceptorDensityProfile::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::r:Area hOc1 (V1, 17, CalcS)::e715e1f7-2079-45c4-a67f-f76b102acfce::48ce018be081dafb160287031fbe08c3",  # 16 Receptor Density Profile features
    "e715e1f7-2079-45c4-a67f-f76b102acfce--e7b46dcf4fea599385b5653ab78e9784",  # Receptor Density Profile: alpha4beta2
    "cf0::StreamlineCounts::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::f16e449d-86e1-408b-9487-aa9d72e39901::295693e37131fd55fbcbcac3b35b3f8b",  # 200 Streamline Counts features cohort: HCP
    "f16e449d-86e1-408b-9487-aa9d72e39901--a45c0c5f53325ac32b59833e7605b18a",  # 015 - Streamline Counts cohort: HCP
    "3f179784-194d-4795-9d8d-301b524ca00a--e27e3ad4f467fb5c445a504557f340a4--9c08356ec0454773885ded630e49b5d3",  # 0108_1 - Functional Connectivity cohort: 1000BRAINS, paradigm: Resting state (RestEmpCorrFC)
    "cf0::FunctionalConnectivity::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290::f16e449d-86e1-408b-9487-aa9d72e39901::4da1dae86a1fd717e5a3618ab041fd3f",  # 200 Functional Connectivity features cohort: HCP, paradigm: Resting state (EmpCorrFC concatenated)
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
    # employ semantic links between regions, potentially changing the result.
    region = siibra.get_region("julich 2.9", "ca1")
    volume = region.get_regional_mask("mni152")
    profiles_region = siibra.features.get(region, "BigBrainIntensityProfile")[0]
    profiles_volume = siibra.features.get(volume, "BigBrainIntensityProfile")[0]
    # the ids will be different but the content has to be the same. Even the order.
    assert [p.location for p in profiles_region] == [
        p.location for p in profiles_volume
    ]


WagstylProfileLoader._load()
verts = WagstylProfileLoader._vertices
np.random.seed(10)
loc_queries = [
    (
        siibra.get_map("julich 2.9", "bigbrain")
        .get_volume("hoc1 left")
        .get_boundingbox(clip=True),
        "BigBrainIntensityProfile",
        13968,
    ),
    (
        PointCloud(verts[np.random.randint(0, len(verts) - 1, 100)], space="bigbrain"),
        "BigBrainIntensityProfile",
        100,
    ),
    (
        PointCloud(
            np.concatenate(
                [
                    verts[np.random.randint(0, len(verts) - 1, 100)] + 1000,
                    verts[np.random.randint(0, len(verts) - 1, 10)],
                ]
            ),
            space="bigbrain",
        ),
        "BigBrainIntensityProfile",
        10,
    ),
]


@pytest.mark.parametrize("loc, feat_type, res_len", loc_queries)
def test_query_w_locations(loc: Location, feat_type: str, res_len: int):
    results = siibra.features.get(loc, feat_type)
    assert len(results) > 0
    assert results[0].data is not None
    if len(results) == 1 and isinstance(results[0], CompoundFeature):
        assert len(results[0]) == res_len
    else:
        assert len(results) == res_len
