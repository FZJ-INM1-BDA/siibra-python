import pathlib
import runpy
import pytest
import sys
from e2e.features.molecular.test_genes import xfail_if_allen_api_unavailable

examples = pathlib.Path('./examples').resolve().rglob('*.py')
python_version = sys.version_info


@pytest.mark.parametrize('example', examples)
@xfail_if_allen_api_unavailable
def test_script_execution(example: pathlib.Path):
    if example.name == "test_examples.py":
        print("Skipping:", example)
        return
    print("Running:", example)
    runpy.run_path(example)
