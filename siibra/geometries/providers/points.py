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
"""Providers of points from tables, served memory-mapped from the cache."""

from __future__ import annotations

from abc import abstractmethod
import json
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .provider import GeometryProvider, _convert_once, _local_file, _source_key

PointArrays = Tuple[np.ndarray, Optional[np.ndarray]]
"""Coordinates as an (N, 3) array, and per-point attributes as a structured array, if any."""


def _save_npy(path: str, array: np.ndarray):
    with open(path, "wb") as f:  # a file object, so that numpy does not append ".npy"
        np.save(f, array, allow_pickle=False)


def _load_npy(path: str) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r")
    except ValueError:  # an empty array cannot be memory-mapped
        return np.load(path)


class _PointTableProvider(GeometryProvider):
    """
    Points read from one table per fragment, with columns x, y and z (case-
    insensitive) in mm and any further columns as per-point attributes.

    Each table is parsed once and stored in siibra's cache as two `.npy`
    files, coordinates and attributes, which are then memory-mapped. Text
    formats cannot be memory-mapped themselves, since their values have no
    fixed position in the file.
    """

    COORDINATE_COLUMNS = ("x", "y", "z")

    def __init__(self, src: Union[str, Dict[str, str]]):
        """
        Parameters
        ----------
        src: str or dict
            URL or local path of a table, or a mapping of fragment names to such sources.
        """
        if isinstance(src, str):
            self._sources: Dict[Optional[str], str] = {None: src}
        elif isinstance(src, dict) and len(src) > 0:
            self._sources = dict(src)
        else:
            raise ValueError(
                f"A {self.srctype} provider takes a URL or path, or a mapping of fragments to them, got {src!r}."
            )
        self._src = src

    @property
    def _url(self) -> Union[str, Dict[str, str]]:
        return self._src

    @property
    def fragments(self) -> List[str]:
        return [name for name in self._sources if name is not None]

    @abstractmethod
    def _read(self, path: str) -> pd.DataFrame:
        """Read a local source file into a table."""
        raise NotImplementedError

    def _split(self, table: pd.DataFrame, src: str) -> PointArrays:
        """Split a table into coordinates and a structured array of the remaining columns."""
        columns = {str(c).lower(): c for c in table.columns}
        missing = [c for c in self.COORDINATE_COLUMNS if c not in columns]
        if missing:
            raise ValueError(
                f"{src} has no column(s) {missing}. Point tables need x, y and z "
                f"columns in mm; found {list(table.columns)}."
            )
        xyz = [columns[c] for c in self.COORDINATE_COLUMNS]
        coordinates = np.ascontiguousarray(table[xyz].to_numpy(dtype=float))
        others = [c for c in table.columns if c not in xyz]
        if not others:
            return coordinates, np.zeros(0)  # stored as a marker for "no attributes"
        fields = []
        for column in others:
            values = table[column].to_numpy()
            if values.dtype == object:  # strings etc.: fixed-width unicode, which needs no pickling
                values = values.astype(str)
            fields.append((str(column), values))
        attributes = np.empty(len(table), dtype=[(name, values.dtype) for name, values in fields])
        for name, values in fields:
            attributes[name] = values
        return coordinates, attributes

    def _arrays(self, fragment: Optional[str]) -> PointArrays:
        """The memory-mapped arrays of one fragment, converting its source if needed."""
        src = self._sources[fragment]
        key = json.dumps([self.srctype, _source_key(src)])
        parsed: Dict[str, PointArrays] = {}

        def parse() -> PointArrays:  # parse at most once, for whichever file is missing
            if "arrays" not in parsed:
                parsed["arrays"] = self._split(self._read(_local_file(src)), src)
            return parsed["arrays"]

        def writer(index: int) -> Callable[[str], None]:
            return lambda path: _save_npy(path, parse()[index])

        coordinates = _load_npy(_convert_once(f"{key}#coordinates", ".npy", writer(0)))
        attributes = _load_npy(_convert_once(f"{key}#attributes", ".npy", writer(1)))
        return coordinates, (attributes if attributes.dtype.names else None)

    def _keys(self, fragments: Optional[List[str]]) -> List[Optional[str]]:
        if fragments is None:
            return list(self._sources)
        unknown = [f for f in fragments if f not in self._sources]
        if unknown:
            raise ValueError(f"No fragment(s) {unknown} in {self._url}.")
        return list(fragments)

    def fetch(self, fragments: List[str] = None) -> Dict[Optional[str], PointArrays]:
        """
        Return the coordinates and attributes of each fragment, by fragment
        name. An unfragmented source has the single key None.
        """
        return {name: self._arrays(name) for name in self._keys(fragments)}

    def get_coordinates(self, fragments: List[str] = None) -> np.ndarray:
        """Coordinates as stored; memory-mapped when from a single source."""
        arrays = [self._arrays(name)[0] for name in self._keys(fragments)]
        return arrays[0] if len(arrays) == 1 else np.concatenate(arrays)


class TsvPointsProvider(_PointTableProvider, srctype="tsv"):
    """Points from tab-separated tables."""

    def _read(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path, sep="\t")


class CsvPointsProvider(_PointTableProvider, srctype="csv"):
    """Points from comma-separated tables."""

    def _read(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)


class NpyPointsProvider(_PointTableProvider, srctype="npy"):
    """Points from `.npy` files: an (N, 3) array, or a structured array with x, y and z fields."""

    def _read(self, path: str) -> pd.DataFrame:
        array = np.load(path, allow_pickle=False)
        if array.dtype.names:
            return pd.DataFrame(array)
        if array.ndim == 2 and array.shape[1] == 3:
            return pd.DataFrame(array, columns=list(self.COORDINATE_COLUMNS))
        raise ValueError(
            f"{path} holds an array of shape {array.shape}; expected (N, 3) "
            "or a structured array with x, y and z fields."
        )
