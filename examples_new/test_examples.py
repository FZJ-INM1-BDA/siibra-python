import pathlib
import runpy
import pytest
import sys

examples = pathlib.Path('./examples_new').resolve().rglob('*.py')
python_version = sys.version_info


@pytest.mark.parametrize('example', examples)
def test_script_execution(example: pathlib.Path):
    if example.name == "test_examples.py":
        print("Skipping:", example)
        return
    print("Running:", example)
    runpy.run_path(example)
