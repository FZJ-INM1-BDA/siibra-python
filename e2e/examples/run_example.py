
import runpy
import sys

from e2e.util import ALLEN_UNAVAILABLE_EXIT_CODE, is_allen_api_unavailable_exception


def main(path: str):
    try:
        runpy.run_path(path, run_name="__main__")
    except Exception as e:
        if is_allen_api_unavailable_exception(e):
            print(repr(e), file=sys.stderr)
            sys.exit(ALLEN_UNAVAILABLE_EXIT_CODE)
        raise e


if __name__ == "__main__":
    main(sys.argv[1])
