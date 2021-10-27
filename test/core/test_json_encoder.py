from typing import Optional
try:
    from typing_extensions import TypedDict
except:
    from typing import TypedDict
import pytest
from siibra.core.json_encoder import CircularReferenceException, JSONEncoder, JSONableConcept, UnJSONableException

nested_flag=True

@pytest.mark.parametrize('primitive_input', [ 'hello wolrd', 1, 1.5 ])
def test_primitive(primitive_input):
    with JSONEncoder(nested=nested_flag) as handle:
        assert handle.get_json(primitive_input) is primitive_input


@pytest.mark.parametrize('list_input, expected_output', [
    ([1,2,3], [1,2,3]),
    ((1,2,3), [1,2,3])
])
def test_list(list_input, expected_output):
    with JSONEncoder(nested=nested_flag) as handle:
        assert handle.get_json(list_input) == expected_output


def test_circular_ref():
    a=['1', '2']
    b=['3', '4', a]
    a.append(b)
    with pytest.raises(CircularReferenceException):
        with JSONEncoder(nested=nested_flag) as handle:
            handle.get_json(a)


def test_dictionary():
    a={
        'hello': 'world',
        'foo': ['bar', 'baz']
    }
    with JSONEncoder(nested=nested_flag) as handle:
        assert handle.get_json(a) == a


def test_circular_ref_dictionary():
    a={}
    b={
        'foo': a
    }
    a['bar'] = b
    
    with pytest.raises(CircularReferenceException):
        with JSONEncoder() as handle:
            handle.get_json(a)

def test_empty_list():
    empty_list=[]
    a={
        'foo': empty_list,
        'bar': empty_list,
        'baz': empty_list,
        'wok': empty_list
    }
    with JSONEncoder(nested=nested_flag) as handle:
        js = handle.get_json(a)
        assert js.get('foo') == []
        assert js.get('bar') == []
        assert js.get('baz') == []
        assert js.get('wok') == []

class DummyCls3:
    pass


class DummyCls2(JSONableConcept):
    typed_json_output = TypedDict('T', {
        'id': str
    })
    def to_json(self, **kwargs):
        return {
            'id': 'dummy-cls-2',
        }

    def from_json(self):
        pass


class DummyCls(JSONableConcept):
    typed_json_output = TypedDict('T', {
        'id': str,
        'child': Optional[any]
    })
    def __init__(self, child=None):
        self.child = child

    def to_json(self, detail=False, **kwargs):
        detail_info={
            'child': self.child
        } if detail else {}
        return {
            'id': 'id-bar',
            **detail_info
        }

    def from_json(self):
        pass

@pytest.mark.parametrize('input,detail_flag,expect_fail', [
    (DummyCls(), True, False),
    (DummyCls(child=DummyCls2()), True, False),
    (DummyCls(child=DummyCls3()), True, True),
    (DummyCls(child=DummyCls3()), False, False),
])
def test_jsonable_concept_subclass(input,detail_flag,expect_fail):
    with JSONEncoder(nested=nested_flag) as handle:
        if expect_fail:
            with pytest.raises(UnJSONableException):
                handle.get_json(input, detail=detail_flag)
        else:
            result=handle.get_json(input, detail=detail_flag)
            assert result.get('id') == 'id-bar'
            if detail_flag:
                assert 'child' in result
            else:
                assert 'child' not in result

def test_jsonable_concept_not_nested():
    inst = DummyCls(child=DummyCls(child=DummyCls2()))
    with JSONEncoder() as handle:
        output=handle.get_json(inst, detail=True)
        assert 'payload' in output and 'references' in output
        assert list(output.get('payload').get('child').keys()) == ['@id']
        child_id=output.get('payload').get('child').get('@id')
        child_in_ref=[ref for ref in output.get('references', []) if ref['@id'] == child_id]
        assert len(child_in_ref) == 1

        # since depth == 1, the second level child will be empty object
        assert inst.child.child is not None
        assert child_in_ref[0].get('child') == {}