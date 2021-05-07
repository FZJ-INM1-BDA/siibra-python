import unittest
from unittest import mock, TestCase
import os
import importlib
import siibra

class TestConfig1(TestCase):

    def test_exported_variable(self):
        os.environ['SIIBRA_CONFIG_GITLAB_PROJECT_TAG'] = ''
        importlib.reload(siibra)
        importlib.reload(siibra.config)
        self.assertEqual(
            siibra.config.GITLAB_PROJECT_TAG,
            f"siibra-{siibra.__version__}"
        )

    @mock.patch.dict(os.environ, { 'SIIBRA_CONFIG_GITLAB_PROJECT_TAG': 'develop' }, clear=True)
    def test_when_env_set(self):

        importlib.reload(siibra)
        importlib.reload(siibra.config)
        self.assertEqual(
            siibra.config.GITLAB_PROJECT_TAG,
            "develop"
        )

if __name__ == "__main__":
    unittest.main()
