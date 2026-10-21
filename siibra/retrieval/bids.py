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
Sidecar metadata is resolved with the BIDS inheritance principle on a
RepositoryConnector holding a BIDS-structured dataset.

Differences from BIDS
---------------------
- Entity keys are not checked against the BIDS schema, so any key is accepted.
  Index values (``run``, ``echo``, ...) are not checked to be integers.
- With ``strict=False`` (the default), names that break the filename rules are
  still parsed, and every deviation is logged at debug level. Parts that cannot
  be read as an entity, empty keys or values, and repeated entities are
  skipped; entities with characters outside the label format (such as the
  hyphens in ``desc-julich-brain``) are kept, since several datasets siibra
  integrates use such names. ``strict=True`` rejects all of these.
- A suffix is required to match sidecars, as in BIDS. Files without one are
  parsed (``suffix`` is None), but sidecar lookup raises ValueError for them.
- The datatype is only taken from a parent folder named like a BIDS datatype.
- Only JSON sidecars are resolved; other inheritable files (``.tsv``,
  ``.bvec``, ...) are not.
"""

import re
from pathlib import PurePath, PurePosixPath
from typing import (
    TYPE_CHECKING,
    Any,
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

# Entity keys and suffixes are alphanumeric. Entity values follow the BIDS
# label format, which also allows "+".
_ALNUM = re.compile(r"[A-Za-z0-9]+")
_LABEL = re.compile(r"[A-Za-z0-9+]+")


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
        If True, raise ValueError for any name that breaks the BIDS filename
        rules. If False, parse what can be parsed and log each deviation at
        debug level (see the module docstring for what is skipped or kept).

    Returns
    -------
    BIDSFileInfo
        ``entities`` keeps the order of appearance; all values are strings
        (e.g. ``run-01`` stays ``"01"``). ``extension`` includes the leading
        dot and everything after the first dot (``.nii.gz``, ``.dtseries.nii``).
        ``suffix`` is None if the last part is not a suffix.
    """
    path = PurePath(filename)
    stem, dot, rest = path.name.partition(".")
    extension = dot + rest

    entities: Dict[str, str] = {}
    suffix: Optional[str] = None
    parts = stem.split("_") if stem else []

    for i, part in enumerate(parts):
        key, sep, value = part.partition("-")
        is_last = i == len(parts) - 1

        if not sep:
            if is_last and part:
                if not _ALNUM.fullmatch(part):
                    _malformed(f"Non-alphanumeric suffix {part!r} in {path.name!r}", strict)
                suffix = part
            else:
                _malformed(f"Unexpected non key-value part {part!r} in {path.name!r}", strict)
            continue

        if not key or not value:
            _malformed(f"Empty key or value in {part!r} of {path.name!r}", strict)
            continue
        if key in entities:
            _malformed(f"Duplicate entity {key!r} in {path.name!r}", strict)
            continue
        if not (_ALNUM.fullmatch(key) and _LABEL.fullmatch(value)):
            # kept when not strict: see the module docstring
            _malformed(f"Entity {part!r} in {path.name!r} has characters outside the BIDS format", strict)

        entities[key] = value

    if suffix is None:
        _malformed(f"No suffix found in {path.name!r}", strict)

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


def find_bids_sidecars(
    filename: str, repository: "RepositoryConnector"
) -> List[str]:
    """
    Find the JSON sidecars that apply to a file in a BIDS repository.

    Following the BIDS inheritance principle, a JSON file applies if it lives in
    the file's folder or any parent folder up to the dataset root, has the same
    suffix, and its entities are a subset (with equal values) of the file's
    entities. BIDS allows at most one applicable sidecar per folder; if there
    are several, a warning is logged and the one with the most entities is used.

    Parameters
    ----------
    filename : str
        Path of the data file, relative to the repository root.
    repository : RepositoryConnector
        Repository holding the BIDS dataset.

    Returns
    -------
    list of str
        Paths of the applicable sidecars relative to the repository root,
        ordered from the dataset root down to the file's folder, so later
        ones override earlier ones. Empty if none applies.

    Raises
    ------
    ValueError
        If the filename has no suffix, since BIDS matches sidecars by suffix.
    """
    info = parse_bids_filename(filename)
    if info.suffix is None:
        raise ValueError(f"Cannot look up sidecars for {filename!r}: it has no BIDS suffix.")

    own_path = PurePosixPath(filename.strip("/"))
    found_per_level: List[str] = []

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
            found_per_level.append(max(candidates)[1])

        if DATASET_DESCRIPTION in names:
            break

    return found_per_level[::-1]


def load_bids_metadata(
    filename: str, repository: "RepositoryConnector"
) -> Dict[str, Any]:
    """
    Metadata of a file in a BIDS repository, merged from its JSON sidecars.

    Following the BIDS inheritance principle, key-values are read from the
    dataset root downwards, and a key in a lower-level sidecar overrides the
    same key from a higher level. Values are not merged below the top level
    (a nested object is replaced, not combined).

    Parameters
    ----------
    filename : str
        Path of the data file, relative to the repository root.
    repository : RepositoryConnector
        Repository holding the BIDS dataset.

    Returns
    -------
    dict
        The merged metadata; empty if no sidecar applies.
    """
    metadata: Dict[str, Any] = {}
    for path in find_bids_sidecars(filename, repository):
        content = repository.get(path)
        if not isinstance(content, dict):
            raise ValueError(f"Sidecar {path!r} does not contain a JSON object.")
        metadata.update(content)
    return metadata
