from typing import Any, Optional
import pytest
from siibra.core.json_encoder import CircularReferenceException, JSONEncoder, SiibraSerializable, UnJSONableException
from siibra.core.jsonable import SiibraBaseSerialization

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


class TypedCls2Output(SiibraBaseSerialization):
    id: str

class DummyCls2(SiibraSerializable):
    
    SiibraSerializationSchema = TypedCls2Output
    def to_json(self, **kwargs):
        return {
            'data': 'dummy-cls-2-data',
            '@id': 'dummy-cls-2',
            '@type': 'dummy-cls'
        }

    def from_json(self):
        pass


class TypedClsOutput(SiibraBaseSerialization):
    id: str
    child: Optional[Any]

class DummyCls(SiibraSerializable):
    SiibraSerializationSchema=TypedClsOutput
    def __init__(self, child=None):
        self.child = child

    def to_json(self, detail=False, **kwargs):
        detail_info={
            'child': self.child
        } if detail else {}
        return {
            'id': 'id-bar',
            '@id': 'id-bar2',
            '@type': 'dummy-cls',
            **detail_info
        }

    def from_json(self):
        pass

expected_base_nested_output={
    'id': 'id-bar',
    '@id': 'id-bar2',
    '@type': 'dummy-cls',
}
expected_detail_nested_output={
    **expected_base_nested_output,
    'child': {
        'data': 'dummy-cls-2-data',
        '@id': 'dummy-cls-2',
        '@type': 'dummy-cls'
    }
}

@pytest.mark.parametrize('input,detail_flag,expect_fail,expected_output', [
    (DummyCls(), True, False, {**expected_base_nested_output, 'child': None}),
    (DummyCls(child=DummyCls2()), True, False, expected_detail_nested_output),
    (DummyCls(child=DummyCls3()), True, True, None),
    (DummyCls(child=DummyCls3()), False, False, expected_base_nested_output),
])
def test_jsonable_concept_subclass(input,detail_flag,expect_fail, expected_output):
    with JSONEncoder(nested=nested_flag) as handle:
        if expect_fail:
            with pytest.raises(UnJSONableException):
                handle.get_json(input, detail=detail_flag)
        if expected_output:
            result=handle.get_json(input, detail=detail_flag)
            assert expected_output == result

def test_jsonable_concept_not_nested():
    inst = DummyCls(
                child=DummyCls(
                    child=DummyCls2()))

    expected_flat_json={
        'payload': {
            'id': 'id-bar',
            '@id': 'id-bar2',
            '@type': 'dummy-cls',
            'child': {
                '@id': 'id-bar2',
                '@type': 'dummy-cls',
            }
        },
        'references': [{
            '@id': 'dummy-cls-2',
            '@type': 'dummy-cls',
            'data': 'dummy-cls-2-data',
        },{
            'id': 'id-bar',
            '@id': 'id-bar2',
            '@type': 'dummy-cls',
            'child': {
                '@id': 'dummy-cls-2',
                '@type': 'dummy-cls',
            }
        }]
    }

    output = JSONEncoder.encode(inst, detail=True)
    assert output == expected_flat_json

def test_jsonable_concept_not_nested_list():
    inst = [DummyCls(child=DummyCls2()), DummyCls(child=DummyCls2()), DummyCls(child=DummyCls2())]
    with JSONEncoder() as handle:
        output=handle.get_json(inst, detail=True)
        payload=output.get('payload')
        assert len(payload) == 3
        assert all([item.get('@id') is not None for item in payload])
