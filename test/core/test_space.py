import unittest

from siibra.core.space import Space
from uuid import uuid4
from parameterized import parameterized
import inspect

class DummyCls:
    def __init__(self, name) -> None:
        self.name = name


class TestSpaces(unittest.TestCase):

    @staticmethod
    def get_instance(volumes=[]):
        return Space(str(uuid4()), "foo-bar", volumes=volumes)

    @classmethod
    def setUpClass(cls) -> None:
        cls.space = TestSpaces.get_instance()

    def test_space_init(self):
        self.assertIsNotNone(self.space)
        self.assertIsInstance(self.space, Space)
    

    @parameterized.expand([
        ([ DummyCls("foo"), DummyCls("bar"), DummyCls("bar") ], None, 0),
        ([ DummyCls("foo"), DummyCls("bar"), DummyCls("bar") ], "foo", 0),
        ([ DummyCls("foo"), DummyCls("bar"), DummyCls("bar") ], "bar", 1),
        ([ DummyCls("foo"), DummyCls("bar"), DummyCls("bar") ], "baz", AssertionError),
    ])
    def test_space_get_template(self, volumes, variant, result_idx):
        self.space = TestSpaces.get_instance(volumes=volumes)
        if inspect.isclass(result_idx) and issubclass(result_idx, Exception):
            with self.assertRaises(result_idx):
                self.space.get_template(variant)
            return
        actual_result = self.space.get_template(variant)
        self.assertIs(actual_result, volumes[result_idx])
