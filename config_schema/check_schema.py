"""
Usage: python config_schema/check_schema.py <path_to_siibra_config>
"""

import os
import json
from enum import Enum
from jsonschema import validate, ValidationError, RefResolver
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from urllib.parse import urljoin
import sys
from itertools import repeat

use_thread = True

skip_path = ("snapshots/ebrainsquery/v1", "venv")

skip_types = []


class ValidationResult(Enum):
    SKIPPED = "SKIPPED"
    PASSED = "PASSED"
    FAILED = "FAILED"


BASE_URI = "http://example.com"

ROOT_DIR = os.path.abspath(f"{os.path.dirname(os.path.realpath(__file__))}/..")


def get_ref(schema):
    resolver = RefResolver(base_uri=BASE_URI, referrer=schema)
    walk_path = f"{ROOT_DIR}/config_schema"
    for dirpath, dirnames, filenames in os.walk(walk_path):
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            with open(f"{dirpath}/{filename}", "r") as fp:
                relative_path = dirpath.replace(walk_path, "") + "/" + filename

                key0 = urljoin(BASE_URI, f"config_schema/{relative_path}")
                key1 = urljoin(BASE_URI, relative_path)

                schema = json.load(fp)

                resolver.store[key0] = schema
                resolver.store[key1] = schema

    return resolver


def validate_json(path_to_json, fail_fast=False):
    if any([path_fragment in path_to_json for path_fragment in skip_path]):
        return (
            path_to_json,
            ValidationResult.SKIPPED,
            None,
        )
    with open(path_to_json, "r") as fp:
        json_obj = json.load(fp)

    # skip list
    if isinstance(json_obj, list):
        return (path_to_json, ValidationResult.SKIPPED, None)
    _type = json_obj.get("@type", None)
    if not _type:
        # TODO consolidate how error are raied
        if fail_fast:
            raise ValidationError(f"type does not exist: {path_to_json}")
        return (path_to_json, ValidationResult.FAILED, None)

    # assert _schema is None
    if not _type or not _type.startswith("siibra"):
        return (path_to_json, ValidationResult.SKIPPED, None)
    if _type in skip_types:
        return (path_to_json, ValidationResult.SKIPPED, None)
    abspath = os.path.join(ROOT_DIR, "config_schema", (_type + ".json"))
    path_to_schema = os.path.abspath(abspath)
    with open(path_to_schema, "r") as fp:
        schema = json.load(fp)
    try:
        resolver = get_ref(schema)
        validate(json_obj, schema, resolver=resolver)
    except ValidationError as e:
        if fail_fast:
            # TODO consolidate how error are raied
            raise e
        return (path_to_json, ValidationResult.FAILED, e)
    return (path_to_json, ValidationResult.PASSED, None)


def main(path_to_configuration: str, *args):
    json_files = [
        f"{dirpath}/{filename}"
        for dirpath, dirnames, filenames in os.walk(path_to_configuration)
        for filename in filenames
        if filename.endswith(".json")
    ]
    # TODO use argparse
    fail_fast = "--fail-fast" in args
    if use_thread:
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = [
                progress
                for progress in tqdm(
                    executor.map(validate_json, json_files, repeat(fail_fast)),
                    total=len(json_files),
                )
            ]
    else:
        result = [validate_json(f, fail_fast) for f in json_files]

    passed = [r for r in result if r[1] == ValidationResult.PASSED]
    failed = [r for r in result if r[1] == ValidationResult.FAILED]
    skipped = [r for r in result if r[1] == ValidationResult.SKIPPED]
    print(
        f"Validation results: PASSED: {len(passed)} SKIPPED: {len(skipped)} FAILED: {len(failed)}"
    )

    if len(failed) > 0:
        print(failed)
        # TODO consolidate how error are raied
        raise ValidationError(
            message="\n-----\n".join([f"{f[0]}: {str(f[2])}" for f in failed])
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        raise RuntimeError("Need path to configuration directory")
    main(*args)
