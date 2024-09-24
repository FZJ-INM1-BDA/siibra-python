import pytest
from unittest.mock import MagicMock, patch, Mock
from siibra.operations.base import DataOp, Merge
import siibra


@pytest.fixture(scope="session")
def MockDataOp1():
    class MockDataOp1(DataOp):
        type = "test/op1"

        @staticmethod
        def to_json():
            return {"type": "test/op1"}

    yield MockDataOp1


@pytest.fixture(scope="session")
def MockDataOp2():
    class MockDataOp1(DataOp):
        type = "test/op2"

        @staticmethod
        def to_json():
            return {"type": "test/op2"}

    yield MockDataOp1


@patch.object(siibra.attributes.dataproviders.base, "run_steps")
def test_merge(run_steps_mock: Mock, MockDataOp1, MockDataOp2):
    run_steps_mock.side_effect = [1, 2]
    input1 = [MockDataOp1.to_json()]
    input2 = [MockDataOp2.to_json()]
    spec = Merge.generate_specs(
        srcs=[
            input1,
            input2,
        ]
    )
    assert Merge().run(None, **spec) == [1, 2]
    assert len(run_steps_mock.call_args_list) == 2

    assert run_steps_mock.call_args_list[0].args == (input1,)
    assert run_steps_mock.call_args_list[1].args == (input2,)
