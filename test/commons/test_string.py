import pytest

from siibra.commons_new.string import fuzzy_match, splitstr

splitstr_args = [
    ("Julich-Brain Cytoarchitectonic Atlas (v2.9)", ["Julich-Brain", "Cytoarchitectonic", "Atlas", "v2.9"]),
    ("DiFuMo Atlas (512 dimensions)", ["DiFuMo", "Atlas", "512", "dimensions"]),
]

@pytest.mark.parametrize("input, expected", splitstr_args)
def test_splitstr(input, expected):
    assert splitstr(input) == expected


fuzzy_match_args = [
    ("v2.9", "Julich-Brain Cytoarchitectonic Atlas (v2.9)", True),
    ("2.9", "Julich-Brain Cytoarchitectonic Atlas (v2.9)", True),
    ("v29", "Julich-Brain Cytoarchitectonic Atlas (v2.9)", False),
]

@pytest.mark.parametrize("input, dest, expected", fuzzy_match_args)
def test_fuzzy_match(input, dest, expected):
    assert fuzzy_match(input, dest) == expected
