# Copyright 2018-2026
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Lightweight, dependency-free helpers for BIDS-style filenames and sidecar metadata.

Filenames are parsed generically (key-value pairs + suffix + extension), so
non-core entities such as TemplateFlow's ``tpl-`` or ``atlas-`` are supported.
Sidecar lookup follows the BIDS inheritance principle on a RepositoryConnector
holding a BIDS-structured dataset.
"""

import re
from pathlib import PurePath, PurePosixPath
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    Iterable,
)

from ..commons import logger

if TYPE_CHECKING:
    from .repositories import RepositoryConnector

PathLike = Union[str, PurePath]

BIDS_DATATYPES = frozenset(
    {
        "anat",
        "beh",
        "dwi",
        "eeg",
        "fmap",
        "func",
        "ieeg",
        "meg",
        "micr",
        "motion",
        "mrs",
        "nirs",
        "perf",
        "pet",
    }
)
DATASET_DESCRIPTION = "dataset_description.json"
SIDECAR_EXTENSION = ".json"

_ALNUM = re.compile(r"[A-Za-z0-9]+")


class BIDSFileInfo(NamedTuple):
    """Parsed components of a BIDS-style filename."""

    entities: Dict[str, str]
    suffix: Optional[str]
    extension: str
    datatype: Optional[str]


def _malformed(message: str, strict: bool) -> None:
    if strict:
        raise ValueError(message)
    logger.debug(message)


def parse_bids_filename(filename: PathLike, strict: bool = False) -> BIDSFileInfo:
    """
    Parse a BIDS-style filename into entities, suffix, extension and datatype.

    Parameters
    ----------
    filename : str or PurePath
        A file path or bare filename. Only the last path component is parsed
        for entities; the parent directory name is used to infer the datatype.
    strict : bool, default False
        If True, raise ValueError on malformed parts, non-alphanumeric
        keys/values, duplicate entities, or a missing suffix. If False,
        such parts are skipped (first occurrence wins for duplicates).

    Returns
    -------
    BIDSFileInfo
        ``entities`` keeps the order of appearance; all values are strings
        (e.g. ``run-01`` stays ``"01"``). ``extension`` includes the leading
        dot and everything after the first dot (``.nii.gz``, ``.dtseries.nii``).
    """
    path = PurePath(filename)
    stem, dot, rest = path.name.partition(".")
    extension = dot + rest

    entities: Dict[str, str] = {}
    suffix: Optional[str] = None
    parts = stem.split("_") if stem else []

    for i, part in enumerate(parts):
        key, sep, value = part.partition("-")

        if not sep:
            if i == len(parts) - 1 and part:
                suffix = part
            else:
                _malformed(
                    f"Unexpected non key-value part {part!r} in {path.name!r}", strict
                )
            continue

        if not key or not value:
            _malformed(f"Empty key or value in {part!r} of {path.name!r}", strict)
            continue
        if strict and not (_ALNUM.fullmatch(key) and _ALNUM.fullmatch(value)):
            raise ValueError(f"Non-alphanumeric entity {part!r} in {path.name!r}")
        if key in entities:
            _malformed(f"Duplicate entity {key!r} in {path.name!r}", strict)
            continue

        entities[key] = value

    if strict and suffix is None:
        raise ValueError(f"No suffix found in {path.name!r}")

    parent = path.parent.name
    datatype = parent if parent in BIDS_DATATYPES else None

    return BIDSFileInfo(
        entities=entities, suffix=suffix, extension=extension, datatype=datatype
    )


def _applicable_sidecars(
    info: BIDSFileInfo,
    folder: PurePosixPath,
    names: Iterable[str],
    own_path: PurePosixPath,
) -> List[Tuple[int, str]]:
    """(entity count, path) of every sidecar in ``folder`` that applies to ``info``."""
    applicable = []
    for name in names:
        path = folder / name
        if path == own_path:
            continue  # the input file itself, if it is a .json
        candidate = parse_bids_filename(name)
        if candidate.suffix != info.suffix or candidate.extension != SIDECAR_EXTENSION:
            continue
        if all(info.entities.get(k) == v for k, v in candidate.entities.items()):
            applicable.append((len(candidate.entities), path.as_posix()))
    return applicable


def find_bids_sidecar(
    filename: str, repository: "RepositoryConnector"
) -> Optional[str]:
    """
    Find the JSON sidecar that applies most specifically to a file in a BIDS repository.

    Following the BIDS inheritance principle, a JSON file applies if it lives in
    the file's folder or any parent folder, has the same suffix, and its entities
    are a subset (with equal values) of the file's entities. The closest folder
    wins; within a folder, the candidate with the most entities wins.

    Parameters
    ----------
    filename : str
        Path of the data file, relative to the repository root.
    repository : RepositoryConnector
        Repository holding the BIDS dataset.

    Returns
    -------
    str or None
        Path of the sidecar relative to the repository root, or None if no
        applicable sidecar exists up to the dataset or repository root.
    """
    info = parse_bids_filename(filename)
    if info.suffix is None:
        logger.debug(f"No suffix in {filename!r}; cannot match a sidecar.")
        return None

    own_path = PurePosixPath(filename.strip("/"))

    for folder in own_path.parents:
        # the root comes out of .parents as ".", which has no parts and so becomes ""
        found = repository.search_files(
            folder="/".join(folder.parts), suffix=SIDECAR_EXTENSION, recursive=False
        )
        names = {PurePosixPath(p).name for p in found}

        candidates = _applicable_sidecars(info, folder, names, own_path)
        if candidates:
            if len(candidates) > 1:
                logger.warning(
                    f"Multiple applicable sidecars in {folder.as_posix()!r}: "
                    f"{sorted(p for _, p in candidates)}. BIDS allows at most one per level; "
                    f"using the one with the most entities."
                )
            _, path = max(candidates)
            return path

        if DATASET_DESCRIPTION in names:
            break

    return None
