import pytest
import siibra

selected_ids = [
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
    "cf0::RegionalBOLD::p:minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-300::1f832869-0361-4064-8e60-c729590e1013::2a993ff2650c1a95e9c331e6a33c4325",  # 200 Regional BOLD features cohort: HCP, paradigm: rfMRI_REST1_RL_BOLD
    "1f832869-0361-4064-8e60-c729590e1013--dcb293fdce0c34447ab066e8725727c1",  # 000 - Regional BOLD cohort: HCP, paradigm: rfMRI_REST1_RL_BOLD
]


@pytest.mark.parametrize("fid", selected_ids)
def test_get_instance(fid):
    feat = siibra.features.Feature._get_instance_by_id(fid)
    feat.plot()
    feat.plot(backend='plotly')
