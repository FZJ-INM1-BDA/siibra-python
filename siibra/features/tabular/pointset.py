# Copyright 2018-2021
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

from . import tabular

from .. import anchor as _anchor

from ...commons import logger
from ...locations import pointset, location
from ...retrieval import requests

from typing import Callable, Dict, Union
import pandas as pd
import numpy as np


class PointsetFeature(tabular.Tabular):
    """
    Pointset
    """

    DESCRIPTION = (
        ""
    )

    def __init__(
        self,
        modality: str,
        files: Dict[str, str],
        space_id: str,
        species: str,
        description: str = "",
        datasets: list = [],
        paradigm: str = ""
    ):
        """
        """
        tabular.Tabular.__init__(
            self,
            modality=modality,
            description=description,
            anchor=_anchor.AnatomicalAnchor(
                species=species,
                location=location.WholeBrain(space_id)
            ),
            datasets=datasets,
            data=None  # lazy loading below
        )
        self._files = files
        self._loaders = {subject: requests.HttpRequest(url) for subject, url in files.items()}
        self.paradigm = paradigm
        self._pointset_cached = dict()

    @property
    def space(self):
        return self.anchor.space

    @property
    def subjects(self):
        """
        Returns the subject identifiers for which the points are available.
        """
        return list(self._files.keys())

    def _load_pointset(self, subject: str) -> pointset.PointSet:
        """
        Extract pointset.
        """
        assert subject in self.subjects, f"'{subject}' is not listed in subjects."
        if len(self._pointset_cached) > 0:
            if subject in self._pointset_cached:
                return self._pointset_cached[subject]

        data = self._loaders[subject].get()
        if len(data) == 1:
            data = data[0]
        if isinstance(data, Dict):
            possible_keys = ["triplets", "coordinates", "points"]
            for key in possible_keys:
                vals = data.get(key, None)
                if vals is not None:
                    if isinstance(vals, Dict):
                        vals = vals.values()
                    break

            coordinates = np.asanyarray(
                [c for c in vals], dtype="double"
            )
            try:
                assert coordinates.shape[1] == 3
            except:
                coordinates = coordinates.reshape((coordinates.shape[0] // 3, 3))
            self._pointset_cached[subject] = pointset.PointSet(
                coordinates=coordinates * 0.0025,
                space=self.space
            )
            return self._pointset_cached[subject]
        else:
            raise NotImplementedError("")

    def get_pointset(self, subject: str = None):
        if subject is None:
            if "all" not in self._pointset_cached:
                coords = []
                for s in self.subjects:
                    coords.extend(
                        [p.coordinate for p in self._load_pointset(s).points]
                    )
                self._pointset_cached["all"] = pointset.PointSet(
                    coordinates=coords,
                    space=self.space
                )
            return self._pointset_cached["all"]
        else:
            return self._load_pointset(subject)

    @property
    def subjects(self):
        """
        Returns the subject identifiers for which signal tables are available.
        """
        return list(self._files.keys())

    @property
    def name(self):
        supername = super().name
        return f"{supername} with paradigm {self.paradigm}"

