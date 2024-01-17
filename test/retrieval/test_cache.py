import pytest
from unittest.mock import MagicMock

from siibra.retrieval.cache import WarmupLevel, Warmup, WarmupRegException

dummy_child1 = MagicMock()
dummy_child2 = MagicMock()

dummy1 = MagicMock(return_value=[dummy_child1, dummy_child2])
dummy2 = MagicMock()

dummy_data1 = MagicMock()
dummy_data2 = MagicMock()


@pytest.fixture
def register_dummy1():
    wrapped_fn = Warmup.register_warmup_fn(WarmupLevel.INSTANCE)(dummy1)
    yield wrapped_fn
    Warmup.deregister_warmup_fn(dummy1)
    dummy1.reset_mock()

    dummy_child1.reset_mock()
    dummy_child2.reset_mock()


@pytest.fixture
def register_dummy2():
    wrapped_fn = Warmup.register_warmup_fn(WarmupLevel.INSTANCE)(dummy2)
    yield wrapped_fn
    Warmup.deregister_warmup_fn(dummy2)
    dummy2.reset_mock()


@pytest.fixture
def register_dummy_data1():
    wrapped_fn = Warmup.register_warmup_fn(WarmupLevel.DATA)(dummy_data1)
    yield wrapped_fn
    Warmup.deregister_warmup_fn(dummy_data1)
    dummy_data1.reset_mock()


@pytest.fixture
def register_dummy_data2():
    wrapped_fn = Warmup.register_warmup_fn(WarmupLevel.DATA)(dummy_data2)
    yield wrapped_fn
    Warmup.deregister_warmup_fn(dummy_data2)
    dummy_data2.reset_mock()


@pytest.fixture
def register_all(
    register_dummy1, register_dummy2, register_dummy_data1, register_dummy_data2
):
    yield


def test_register(register_dummy1):
    assert Warmup.is_registered(dummy1)


def test_deregister_original(register_dummy1):
    Warmup.deregister_warmup_fn(dummy1)
    assert not Warmup.is_registered(dummy1)


def test_deregister_wrapped(register_dummy1):
    Warmup.deregister_warmup_fn(register_dummy1)
    assert not Warmup.is_registered(dummy1)


def test_register_multiple():
    wrapped = Warmup.register_warmup_fn()(dummy1)

    with pytest.raises(WarmupRegException):
        Warmup.register_warmup_fn()(dummy1)

    with pytest.raises(WarmupRegException):
        Warmup.register_warmup_fn()(wrapped)

    Warmup.deregister_warmup_fn(dummy1)


def test_register_as_factory():
    wrapped = Warmup.register_warmup_fn(is_factory=True)(dummy1)
    Warmup.warmup()
    dummy1.assert_called_once()
    dummy_child1.assert_called_once()
    dummy_child2.assert_called_once()

    Warmup.deregister_warmup_fn(wrapped)
    dummy1.reset_mock()
    dummy_child1.reset_mock()
    dummy_child2.reset_mock()


def test_register_not_called(register_all):
    dummy1.assert_not_called()
    dummy2.assert_not_called()


def test_register_warmup_called(register_all):
    Warmup.warmup()
    dummy1.assert_called_once()
    dummy2.assert_called_once()

    dummy_data1.assert_not_called()
    dummy_data2.assert_not_called()


def test_register_warmup_called_level(register_all):
    Warmup.warmup(WarmupLevel.INSTANCE)

    dummy1.assert_called_once()
    dummy2.assert_called_once()

    dummy_data1.assert_not_called()
    dummy_data2.assert_not_called()


def test_register_warmup_called_level_high(register_all):
    Warmup.warmup(WarmupLevel.DATA)

    dummy1.assert_called_once()
    dummy2.assert_called_once()

    dummy_data1.assert_called_once()
    dummy_data2.assert_called_once()
