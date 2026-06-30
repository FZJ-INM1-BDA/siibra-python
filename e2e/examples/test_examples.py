import pathlib
import subprocess
import sys

import pytest

from e2e.util import ALLEN_UNAVAILABLE_EXIT_CODE


def get_examples():
    for example in pathlib.Path("./examples").resolve().rglob("*.py"):
        marks = []
        if example.parent.name == "tutorials":
            marks.append(pytest.mark.tutorial)

        yield pytest.param(example, id=example.name, marks=marks)


@pytest.mark.parametrize("example", get_examples())
def test_script_execution(example: pathlib.Path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "e2e.examples.run_example",
            example.as_posix(),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == ALLEN_UNAVAILABLE_EXIT_CODE:
        pytest.xfail(
            f"Skipping {example.name} because the Allen API is unavailable "
            f"or returned an invalid response.\n{result.stderr}"
        )

    assert result.returncode == 0, (
        f"Example {example.name} failed with exit code {result.returncode}\n\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )
