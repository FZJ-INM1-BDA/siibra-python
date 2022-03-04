import pytest
from siibra.features.ieeg import IEEG_Session, IEEG_SessionQuery

ieeg_query = IEEG_SessionQuery()

@pytest.mark.parametrize('ieeg', ieeg_query.features)
def test_ieeg_to_model(ieeg: IEEG_Session):
    ieeg.to_model()
