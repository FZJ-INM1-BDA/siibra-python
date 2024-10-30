import pytest
from unittest.mock import MagicMock, patch, Mock
from siibra.operations.base import DataOp, Merge, get_parameters, update_parameters
import siibra


@pytest.fixture(scope="session")
def MockDataOp1():
    class MockDataOp1(DataOp):
        type = "test/op1"
        desc = "foo"

        @classmethod
        def generate_specs(cls, foo: str = None, **kwargs):
            base = super().generate_specs(**kwargs)
            return {**base, "foo": foo}

    yield MockDataOp1


@pytest.fixture(scope="session")
def MockDataOp2():
    class MockDataOp1(DataOp):
        type = "test/op2"
        desc = "bar"

        @classmethod
        def generate_specs(cls, bar: str = None, **kwargs):
            base = super().generate_specs(**kwargs)
            return {**base, "bar": bar}

    yield MockDataOp1


@pytest.fixture(scope="session")
def merge_op_spec(MockDataOp1, MockDataOp2):
    input1 = [MockDataOp1.generate_specs()]
    input2 = [MockDataOp2.generate_specs()]
    spec = Merge.generate_specs(
        srcs=[
            input1,
            input2,
        ]
    )
    yield spec


@patch.object(siibra.attributes.datarecipes.base, "run_steps")
def test_merge_run(run_steps_mock: Mock, MockDataOp1, MockDataOp2, merge_op_spec):
    run_steps_mock.side_effect = [1, 2]
    input1 = [MockDataOp1.generate_specs()]
    input2 = [MockDataOp2.generate_specs()]
    assert Merge().run(None, **merge_op_spec) == [1, 2]
    assert len(run_steps_mock.call_args_list) == 2

    assert run_steps_mock.call_args_list[0].args == (input1,)
    assert run_steps_mock.call_args_list[1].args == (input2,)


def test_merge_get_param(merge_op_spec):
    param, cls = get_parameters(merge_op_spec)
    assert "foo" in param
    assert "bar" in param


def test_merge_update_param(merge_op_spec):
    assert merge_op_spec["srcs"][0][0]["foo"] == None
    assert merge_op_spec["srcs"][1][0]["bar"] == None

    assert "bar" not in merge_op_spec["srcs"][0][0]
    assert "foo" not in merge_op_spec["srcs"][1][0]

    merge_op_spec = update_parameters(merge_op_spec, foo="foo", bar="bar")

    assert merge_op_spec["srcs"][0][0]["foo"] == "foo"
    assert merge_op_spec["srcs"][1][0]["bar"] == "bar"

    assert "bar" not in merge_op_spec["srcs"][0][0]
    assert "foo" not in merge_op_spec["srcs"][1][0]
