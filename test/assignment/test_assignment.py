from unittest.mock import MagicMock, patch, Mock
import pytest
from itertools import chain, repeat

import siibra.assignment.assignment
from siibra.attributes import AttributeCollection
from siibra.assignment.qualification import Qualification


a_0, a_1 = MagicMock(), MagicMock()
b_0, b_1 = MagicMock(), MagicMock()
a = AttributeCollection(attributes=[a_0, a_1])
b = AttributeCollection(attributes=[b_0, b_1])


@patch("siibra.assignment.assignment.attribute_is_qualifiable")
@patch("siibra.assignment.assignment.attribute_qualify")
def test_qualify_compared_all_false(mock_qualify: Mock, mock_is_qualifyable: Mock):
    mock_qualify.return_value = None
    mock_is_qualifyable.side_effect = chain([True, True], repeat(False))

    assert list(siibra.assignment.assignment.qualify(a, b)) == []

    mock_is_qualifyable.assert_called()
    assert mock_is_qualifyable.call_count == 4

    mock_qualify.assert_any_call(a_0, b_0)
    assert mock_qualify.call_count == 1


@patch("siibra.assignment.assignment.attribute_is_qualifiable")
@patch("siibra.assignment.assignment.attribute_qualify")
def test_qualify_compared_all_false(mock_qualify: Mock, mock_is_qualifyable: Mock):
    mock_qualify.return_value = None
    mock_is_qualifyable.return_value = True

    assert list(siibra.assignment.assignment.qualify(a, b)) == []

    mock_is_qualifyable.assert_called()
    assert mock_is_qualifyable.call_count == 4

    mock_qualify.assert_any_call(a_0, b_0)
    mock_qualify.assert_any_call(a_0, b_1)
    mock_qualify.assert_any_call(a_1, b_0)
    mock_qualify.assert_any_call(a_1, b_1)
    assert mock_qualify.call_count == 4


@patch("siibra.assignment.assignment.attribute_is_qualifiable")
@patch("siibra.assignment.assignment.attribute_qualify")
def test_qualify_compared_first(mock_qualify: Mock, mock_is_qualifyable: Mock):
    mock_qualify.side_effect = [Qualification.EXACT, None, None, None]
    mock_is_qualifyable.return_value = True

    assert next(siibra.assignment.assignment.qualify(a, b)) == (
        a_0,
        b_0,
        Qualification.EXACT,
    )

    mock_qualify.assert_called_once_with(a_0, b_0)
    assert mock_qualify.call_count == 1


@patch("siibra.assignment.assignment.attribute_is_qualifiable")
@patch("siibra.assignment.assignment.attribute_qualify")
def test_qualify_compared_unreg(mock_qualify: Mock, mock_is_qualifyable: Mock):
    UnregisteredAttrCompException = (
        siibra.assignment.assignment.UnregisteredAttrCompException
    )
    mock_qualify.side_effect = UnregisteredAttrCompException
    mock_is_qualifyable.return_value = True

    with pytest.raises(UnregisteredAttrCompException):
        list(siibra.assignment.assignment.qualify(a, b))

    mock_qualify.assert_any_call(a_0, b_0)
    mock_qualify.assert_any_call(a_0, b_1)
    mock_qualify.assert_any_call(a_1, b_0)
    mock_qualify.assert_any_call(a_1, b_1)
    assert mock_qualify.call_count == 4


@patch("siibra.assignment.assignment.attribute_is_qualifiable")
@patch("siibra.assignment.assignment.attribute_qualify")
def test_qualify_compared_some_false(mock_qualify: Mock, mock_is_qualifyable: Mock):
    mock_qualify.side_effect = [None, None, Qualification.EXACT, None, None, None]
    mock_is_qualifyable.return_value = True

    assert next(siibra.assignment.assignment.qualify(a, b)) == (
        a_1,
        b_0,
        Qualification.EXACT,
    )

    mock_qualify.assert_any_call(a_0, b_0)
    mock_qualify.assert_any_call(a_0, b_1)
    mock_qualify.assert_any_call(a_1, b_0)
    assert mock_qualify.call_count == 3


@patch("siibra.assignment.assignment.attribute_is_qualifiable")
@patch("siibra.assignment.assignment.attribute_qualify")
def test_qualify_compared_invalid(mock_qualify: Mock, mock_is_qualifyable: Mock):
    mock_is_qualifyable.return_value = True
    InvalidAttrCompException = siibra.assignment.assignment.InvalidAttrCompException
    mock_qualify.side_effect = InvalidAttrCompException

    assert len(list(siibra.assignment.assignment.qualify(a, b))) == 0

    mock_qualify.assert_called_once_with(a_0, b_0)
    assert mock_qualify.call_count == 1
