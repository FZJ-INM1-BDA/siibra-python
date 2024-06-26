import unittest

from siibra.core.space import Space
from siibra.commons import Species
from uuid import uuid4
from parameterized import parameterized
import inspect


class DummyDataset:
    def __init__(self, name) -> None:
        self.name = name


class DummyCls:
    def __init__(self, name, datasets=[DummyDataset('dataset')]) -> None:
        self.name = name
        self.variant = name
        self.datasets = datasets


class TestSpaces(unittest.TestCase):
    @staticmethod
    def get_instance(volumes=[]):
        return Space(
            str(uuid4()), "foo-bar", volumes=volumes, species=Species.HOMO_SAPIENS
        )

    @classmethod
    def setUpClass(cls) -> None:
        cls.space = TestSpaces.get_instance()

    def test_space_init(self):
        self.assertIsNotNone(self.space)
        self.assertIsInstance(self.space, Space)

    @parameterized.expand(
        [
            ([DummyCls("foo"), DummyCls("bar"), DummyCls("bar")], None, 0),
            ([DummyCls("foo"), DummyCls("bar"), DummyCls("bar")], "foo", 0),
            ([DummyCls("foo"), DummyCls("bar"), DummyCls("bar")], "bar", 1),
            ([DummyCls("foo"), DummyCls("bar"), DummyCls("bar")], "baz", RuntimeError),
        ]
    )
    def test_space_get_template(self, volumes, variant, result_idx):
        self.space = TestSpaces.get_instance(volumes=volumes)
        if inspect.isclass(result_idx) and issubclass(result_idx, Exception):
            with self.assertRaises(result_idx):
                self.space.get_template(variant)
            return
        actual_result = self.space.get_template(variant)
        self.assertIs(actual_result, volumes[result_idx])
