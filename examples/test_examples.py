import pathlib
import subprocess
import pytest
import sys
from e2e.features.molecular.test_genes import xfail_if_allen_api_unavailable

python_version = sys.version_info


def get_examples():
    for example in pathlib.Path("./examples").resolve().rglob("*.py"):
        if example.name == "test_examples.py":
            continue
        yield pytest.param(example, id=example.name)


@pytest.mark.parametrize("example", get_examples())
@xfail_if_allen_api_unavailable
def test_script_execution(example: pathlib.Path):
    result = subprocess.run(
        ["python", example.as_posix()], capture_output=True, text=True
    )
    assert (
        result.returncode == 0
    ), f"Example {example.name} failed with:\n{result.stderr}"
