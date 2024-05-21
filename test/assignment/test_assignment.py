from unittest.mock import MagicMock, patch, Mock, call
import pytest
from itertools import product

import siibra.assignment.assignment
from siibra.concepts import AttributeCollection


a_0, a_1 = MagicMock(), MagicMock()
b_0, b_1 = MagicMock(), MagicMock()
a = AttributeCollection(
    attributes=[a_0, a_1]
)
b = AttributeCollection(
    attributes=[b_0, b_1]
)

@patch("siibra.assignment.assignment.attribute_match")
def test_match_compared_all_false(mock_match: Mock):
    mock_match.return_value = False
    
    assert siibra.assignment.assignment.match(a, b) == False
    
    mock_match.assert_any_call(a_0, b_0)
    mock_match.assert_any_call(a_0, b_1)
    mock_match.assert_any_call(a_1, b_0)
    mock_match.assert_any_call(a_1, b_1)
    assert mock_match.call_count == 4


@patch("siibra.assignment.assignment.attribute_match")
def test_match_compared_first(mock_match: Mock):
    mock_match.return_value = True
    
    assert siibra.assignment.assignment.match(a, b)
    
    mock_match.assert_called_once_with(a_0, b_0)
    assert mock_match.call_count == 1



@patch("siibra.assignment.assignment.attribute_match")
def test_match_compared_unreg(mock_match: Mock):
    UnregisteredAttrCompException = siibra.assignment.assignment.UnregisteredAttrCompException
    mock_match.side_effect = UnregisteredAttrCompException
    
    with pytest.raises(UnregisteredAttrCompException):
        siibra.assignment.assignment.match(a, b)
    
    mock_match.assert_any_call(a_0, b_0)
    mock_match.assert_any_call(a_0, b_1)
    mock_match.assert_any_call(a_1, b_0)
    mock_match.assert_any_call(a_1, b_1)
    assert mock_match.call_count == 4

@patch("siibra.assignment.assignment.attribute_match")
def test_match_compared_some_false(mock_match: Mock):
    mock_match.side_effect = [False, False, True, False, False, False]
    
    assert siibra.assignment.assignment.match(a, b) == True
    
    mock_match.assert_any_call(a_0, b_0)
    mock_match.assert_any_call(a_0, b_1)
    mock_match.assert_any_call(a_1, b_0)
    assert mock_match.call_count == 3

@patch("siibra.assignment.assignment.attribute_match")
def test_match_compared_invalid(mock_match: Mock):
    InvalidAttrCompException = siibra.assignment.assignment.InvalidAttrCompException
    mock_match.side_effect = InvalidAttrCompException
    
    assert not siibra.assignment.assignment.match(a, b)
    
    mock_match.assert_called_once_with(a_0, b_0)
    assert mock_match.call_count == 1

