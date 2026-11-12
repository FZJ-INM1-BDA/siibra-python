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
"""Live query filtering streamline fiber bundles by the regions they terminate in."""

from pathlib import PurePosixPath
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .query import LiveQuery
from ..commons import logger
from ..core.assignment import Qualification
from ..core.region import Region
from ..features.anchor import AnatomicalAnchor
from ..features.connectivity.streamline_fiber_bundles import StreamlineFiberBundles
from ..retrieval.repositories import DataProxyConnector, RepositoryConnector
from ..retrieval.requests import find_suitable_decoder
from ..volumes.volume import Volume


class BundleFilterQuery(LiveQuery, args=[], FeatureType=StreamlineFiberBundles):
    """
    Finds streamline fiber bundles by the regions they terminate in.

    - A region or parcellation yields, for each preconfigured tractogram, a
      feature with only the bundles terminating in the query's leaf regions,
      anchored to the query concept, which the bundles overlap.
    - A space yields the preconfigured tractograms in that space, with all
      their bundles.

    Terminations are read from tables in a data-proxy bucket, with one row
    per bundle, named in the first column as the TRX group, one column per
    region name, and 1 where the bundle terminates in the region. Region
    names are matched exactly. The tables of a tractogram are in the folder
    ``terminal_region_tables/<modality folder>/``, where the folder is named
    after the modality (see :meth:`modality_folder`). The table for a
    parcellation has the parcellation's key between underscores in its file
    name, e.g. ``MNI152_JBA31_LABELLED_connectivity_matrix_long_bundles.tsv``.
    """

    BUCKET = "ebrains2-t13-chenonceau-hi-res-dmri-bund"
    TABLE_FOLDER = "terminal_region_tables"
    # parcellations with termination tables, by id, with the key in their table names
    PARCELLATION_KEYS = {
        "minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-310": "JBA31",
    }
    MODALITY_FOLDERS = {
        "Streamline long fiber bundles": "long_bundles_in_MNI",
    }

    _connector_cached: Optional[RepositoryConnector] = None
    _tables: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}

    def __init__(self, **kwargs):
        LiveQuery.__init__(self, **kwargs)

    @classmethod
    def _connector(cls) -> RepositoryConnector:
        if cls._connector_cached is None:
            cls._connector_cached = DataProxyConnector(cls.BUCKET)
        return cls._connector_cached

    @classmethod
    def _table(cls, modality: str, parcellation_key: str) -> Optional[pd.DataFrame]:
        """The termination table of a modality and parcellation, read once, or None."""
        folder = f"dMRI_streamline_tracks/{cls.MODALITY_FOLDERS[modality]}/{cls.TABLE_FOLDER}"
        if (folder, parcellation_key) not in cls._tables:
            table = None
            try:
                connector = cls._connector()
                files = [
                    f for f in connector.search_files(folder=f"{folder}/", suffix=".tsv", recursive=True)
                    if parcellation_key in PurePosixPath(f).stem.split("_")
                ]
                if len(files) == 1:
                    table = connector.get_loader(files[0], decode_func=find_suitable_decoder(files[0])).get()
                    table = table.set_index(table.columns[0])
                    table.index = table.index.astype(str)
                elif len(files) > 1:
                    logger.warning(f"Several termination tables for {parcellation_key} in {folder}, using none: {files}")
                else:
                    logger.debug(f"No termination table for {parcellation_key} in {folder}")
            except Exception as e:
                logger.warning(f"Cannot read termination tables from {folder}: {e}")
            cls._tables[folder, parcellation_key] = table
        return cls._tables[folder, parcellation_key]

    def query(self, concept, **kwargs) -> List[StreamlineFiberBundles]:
        if isinstance(concept, Volume):  # a space query passes the space's template
            return [
                f for f in StreamlineFiberBundles._get_instances()
                if f.space is not None and f.space.id == concept.space.id
            ]
        if not isinstance(concept, Region):
            return []

        parcellation_key = self.PARCELLATION_KEYS.get(concept.root.id)
        if parcellation_key is None:
            logger.debug(f"No termination tables for the parcellation of {concept}")
            return []
        leaves = [concept] if concept.is_leaf else list(concept.leaves)
        region_names = {leaf.name for leaf in leaves}

        results = []
        for tractogram in StreamlineFiberBundles._get_instances():
            table = self._table(tractogram.modality, parcellation_key)
            if table is None:
                continue
            columns = [c for c in table.columns if c in region_names]
            if not columns:
                continue
            terminating = table.index[(table[columns] == 1).any(axis=1)]
            if len(terminating) == 0:
                continue
            anchor = AnatomicalAnchor(
                species=tractogram.anchor.species,
                region={concept: Qualification.OVERLAPS},  # the bundles extend beyond the region
            )
            results.append(tractogram._filter_by_bundles(list(terminating), anchor))
        return results
