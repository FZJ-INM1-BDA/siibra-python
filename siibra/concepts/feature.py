# Copyright 2018-2024
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

from typing import List
import pandas as pd

from ..attributes.locations import Location
from ..attributes.descriptions import Modality
from ..attributes.attribute_collection import AttributeCollection, MATRIX_INDEX_ENTITY_KEY, attr_of_general_interest


class Feature(AttributeCollection):
    schema: str = "siibra/concepts/feature/v0.2"


    @property
    def modalities(self):
        return self._find(Modality)

    @property
    def locations(self):
        return self._find(Location)

    @property
    def matrix_indices(self):
        attr = [
            attr for attr in self.attributes if MATRIX_INDEX_ENTITY_KEY in attr.extra
        ]
        return sorted(attr, key=lambda a: a.extra[MATRIX_INDEX_ENTITY_KEY])

    @property
    def data(self):
        from ..attributes.dataitems import Tabular

        matrix_entity_key = self.filter(attr_of_general_interest)

        dfs: List[pd.DataFrame] = [d.get_data() for d in self._find(Tabular)]
        if len(matrix_entity_key.attributes) > 0:

            mapping_idx = {
                attr.extra[MATRIX_INDEX_ENTITY_KEY]: attr
                for attr in matrix_entity_key.attributes
            }

            def remapper(index: int):
                return mapping_idx.get(index, index)

            for df in dfs:
                df.rename(index=remapper, columns=remapper, inplace=True)

        return dfs

    def plot(self, *args, **kwargs):
        from ..attributes.dataitems import Tabular

        return [d.plot(*args, **kwargs) for d in self._find(Tabular)]
