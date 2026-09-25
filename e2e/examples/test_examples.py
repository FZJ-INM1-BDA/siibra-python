import pathlib
import subprocess
import sys
from dataclasses import dataclass

import pytest

from e2e.util import ALLEN_UNAVAILABLE_EXIT_CODE

TAIL_LINES = 40


@dataclass
class ExampleFailure:
    example: pathlib.Path
    returncode: int
    stdout: str
    stderr: str

    @property
    def error(self) -> str:
        """The exception line, i.e. the last unindented line of the traceback."""
        lines = [
            line for line in self.stderr.strip().splitlines()
            if line and not line[0].isspace()
        ]
        return lines[-1] if lines else "(no traceback on stderr)"

    @staticmethod
    def _tail(stream: str) -> str:
        lines = stream.strip().splitlines()
        omitted = len(lines) - TAIL_LINES
        if omitted > 0:
            lines = [f"[... {omitted} earlier line(s) omitted ...]", *lines[-TAIL_LINES:]]
        return "\n".join(lines) or "(empty)"

    def __str__(self) -> str:
        return "\n".join([
            f"{self.example.name} exited with code {self.returncode}",
            f"  {self.error}",
            "",
            f"reproduce with:  python -m e2e.examples.run_example {self.example.as_posix()}",
            "",
            f"--- stderr (last {TAIL_LINES} lines) ---",
            self._tail(self.stderr),
            "",
            f"--- stdout (last {TAIL_LINES} lines) ---",
            self._tail(self.stdout),
        ])


def get_examples():
    return [
        pytest.param(example, id=example.name)
        for example in sorted(pathlib.Path("./examples").resolve().rglob("*.py"))
    ]


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

    if result.returncode != 0:
        pytest.fail(
            str(ExampleFailure(example, result.returncode, result.stdout, result.stderr)),
            pytrace=False,
        )
