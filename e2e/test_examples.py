import pathlib
import runpy
import pytest
import sys

examples = pathlib.Path('./examples').resolve().rglob('*.py')
python_version = sys.version_info


@pytest.mark.skipif(
    (python_version.major, python_version.minor) == (3, 7),
    reason="Examples are expensive, only test with the oldest suported version."
)
@pytest.mark.parametrize('example', examples)
def test_script_execution(example):
    runpy.run_path(example)
