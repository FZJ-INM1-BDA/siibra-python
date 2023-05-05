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
from ...locations import point, pointset, location
from ...retrieval import requests

from typing import Callable, Dict, List, Tuple
from pandas import DataFrame, concat

class PointCloud(
    tabular.Tabular,
    configuration_folder="features/tabular/pointcloud",
    category="molecular"
):
    """
    PointCloud features are comprimised of 3D coordinates from the same
    reference space, and possibly contain additional data such as intensity.
    """

    def __init__(
        self,
        files: Dict[str, str],
        space_spec: str,
        species: str,
        decode_func: Callable,
        description: str = "",
        datasets: list = [],
        modality: str = "",
        paradigm: str = ""
    ):
        """
        Construct a PointCloud dataframe.
        """
        tabular.Tabular.__init__(
            self,
            modality=modality,
            description=description,
            anchor=_anchor.AnatomicalAnchor(
                species=species,
                location=location.WholeBrain(space_spec)
            ),
            datasets=datasets,
            data=None  # lazy loading below
        )
        self._decode_func = decode_func
        self._files = files
        self._loaders = {
            subject: requests.HttpRequest(url, func=self._decode_func)
            for subject, url in files.items()
        }
        self.paradigm = paradigm
        self._pointcloud_cache = dict()

    @property
    def space(self):
        return self.anchor.space

    @property
    def subjects(self):
        """Returns the subject identifiers for which the data is available."""
        return list(self._files.keys())

    @property
    def name(self):
        supername = super().name
        return supername + f" with paradigm {self.paradigm}" if self.paradigm else ""

    def _load_pointcloud_data(self, subject: str):
        """
        Extract coordinates and additional values for a subject.
        """
        assert subject in self.subjects, f"'{subject}' is not listed in {self.subjects}."
        if self._pointcloud_cache.get(subject) is not None:
            return self._pointcloud_cache[subject]
        array = self._loaders[subject].get().values
        coordinates = []
        values = {}
        assert array.shape[1] >= 3
        if array.shape[1] > 3:
            values = {f"value {n}": [] for n in range(array.shape[0] - 3)}
        for line in array:
            coordinates.append(tuple(line[0:3]))
            if len(values) > 0:
                for n in range(array.shape[0] - 3):
                    values[f"value {n}"].append(line[n+3])
        self._pointcloud_cache[subject] = {"Coordinates": coordinates, **values}
        return self._pointcloud_cache[subject]

    def get_table(self, subject: str = None, value_headers: List[str] = []) -> DataFrame:
        """
        Get pandas dataframe with points and their respective additional values.

        Parameters
        ----------
        subject: str, default=None
            List all the subjects by PointCloud.subjects. If subject is None,
            it will return a concatanated dataframe of all subjects.
        value_headers: List[str]
            Headers of the additional data for each point. If not specified,
            'value <number>' is used.
        
        Returns
        -------
        Dataframe
            Coordinates of the points as tuples and additianl values in the data
            set.
        """
        if (subject is None) and (len(self.subjects) > 1):
            logger.info("No subject is selected. Returning all in one dataframe.")
            return concat(
                [self.get_table(s) for s in self.subjects],
                ignore_index=True
            )

        data = self._load_pointcloud_data(subject)
        if len(value_headers) == 0:
            headers = [col_name for col_name in data.keys()]
        else:
            headers = ["Coordinates"].extend(value_headers)
        return DataFrame(data, columns=headers)

    def get_pointset(self, subject: str) -> pointset.PointSet:
        """
        Get PointSet object of the point cloud data for a specified subject.

        PointSet is a useful siibra concept for multiple points. Could be used
        to warp the points to different spaces, get a BoundingBox, find
        centroids, and check intersection with masks or BoundingBox's.

        Parameters
        ----------
        subject: str
            List all the subjects by PointCloud.subjects.
        
        Returns
        -------
        PointSet

        Raises
        ------
        TypeError
            If no subject is specified.
        """
        if (subject is None) and (len(self.subjects) > 1):
            raise TypeError("Missing argument: 'subject'")
        data = self._load_pointcloud_data(subject)
        return pointset.PointSet(data.get("Coordinates"), space=self.space)

    

