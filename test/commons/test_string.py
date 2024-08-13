import pytest
import re

from siibra.commons.string import fuzzy_match, splitstr, get_spec

splitstr_args = [
    (
        "Julich-Brain Cytoarchitectonic Atlas (v2.9)",
        ["Julich", "Brain", "Cytoarchitectonic", "Atlas", "v2.9"],
    ),
    ("DiFuMo Atlas (512 dimensions)", ["DiFuMo", "Atlas", "512", "dimensions"]),
]


@pytest.mark.parametrize("input, expected", splitstr_args)
def test_splitstr(input, expected):
    assert splitstr(input) == expected


fuzzy_match_args = [
    ("julich 3", "Julich-Brain v3.0.3", False),
    ("julich 3.0", "Julich-Brain v3.0.3", False),
    ("julich 3.0.3", "Julich-Brain v3.0.3", True),
    ("julich 9", "Julich-Brain Cytoarchitectonic Atlas (v2.9)", False),
    ("v2.9", "Julich-Brain Cytoarchitectonic Atlas (v2.9)", True),
    ("2.9", "Julich-Brain Cytoarchitectonic Atlas (v2.9)", True),
    ("julich 2.9", "Julich-Brain Cytoarchitectonic Atlas (v2.9)", True),
    ("v29", "Julich-Brain Cytoarchitectonic Atlas (v2.9)", False),
]


@pytest.mark.parametrize("input, dest, expected", fuzzy_match_args)
def test_fuzzy_match(input, dest, expected):
    assert fuzzy_match(input, dest) == expected


@pytest.mark.parametrize("spec, input, expected", [
    ("foo", "foo", True),
    ("foo", "Foo", True),
    ("foo", "foo bar", True),
    ("foo", "Foo Bar", True),
    ("foo", "Food", False),
    ("foo", "food", False),

    (re.compile(r"hello[0-9]"), "hello0", True),
    (re.compile(r"hello[0-9]"), "hello1", True),
    (re.compile(r"hello[0-9]"), "hello10", True),
    (re.compile(r"hello[0-9]"), "hello", False),

    ("/hello[0-9]/", "hello0", True),
    ("/hello[0-9]/", "hello1", True),
    ("/hello[0-9]/", "hello10", True),
    ("/hello[0-9]/", "hello", False),
    
    ("/hello/i", "Hello0", True),
    ("/hello/i", "hEllo1", True),
    ("/hello/", "heLLo10", False),

    ("/^hello/", "hello", True),
    ("/^hello/", "foohello", False),
])
def test_get_spec(spec, input, expected):
    assert bool(get_spec(spec)(input)) == expected
