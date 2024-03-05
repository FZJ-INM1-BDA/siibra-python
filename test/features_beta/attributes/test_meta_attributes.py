import pytest
from itertools import repeat, chain

from siibra.features_beta.attributes import (
    ModalityAttribute,
    NameAttribute,
    Attribute,
)

should_match = chain(
    zip(
        repeat(True),
        repeat(NameAttribute("Hello World is my name")),
        [
            {"name": "hello world"},
            {"name": "Hello World"},
            {"name": "IS"},
            {"buzz": None},
        ]
    ),
    zip(
        repeat(True),
        repeat(ModalityAttribute("RECEPTOR DENSITY PROFILE")),
        [
            {"modality": "receptor"},
            {"modality": "RECEPTOR"},
            {"modality": "rec"},
            {"buzz": None},
        ]
    )
)


should_not_match = chain(
    zip(
        repeat(False),
        repeat(NameAttribute("Hello World is my name")),
        [
            {"name": "foo bar"},
            {"name": "is your name"},
            {"name": "ARE"},
        ]
    ),
    zip(
        repeat(False),
        repeat(ModalityAttribute("RECEPTOR DENSITY PROFILE")),
        [
            {"modality": "foo bar"},
            {"modality": "big brain"},
        ]
    )
)


@pytest.mark.parametrize("should_match, attr, kwargs", chain(
        should_match,
        should_not_match
))
def test_modality_attributes(should_match: bool, attr: Attribute, kwargs):
    assert should_match == attr.filter(**kwargs)


