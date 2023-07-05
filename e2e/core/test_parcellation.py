import pytest
import siibra

all_parcellations = [
    s for s in siibra.parcellations
]

@pytest.mark.parametrize('parcellation', all_parcellations)
def test_has_desc(parcellation: siibra.core.parcellation.Parcellation):
    assert parcellation.description, f"{parcellation.name!r} does not have desc"

@pytest.mark.parametrize('parcellation', all_parcellations)
def test_has_publications(parcellation: siibra.core.parcellation.Parcellation):
    assert len(parcellation.publications) > 0, f"{parcellation.name!r} does not have publication"
