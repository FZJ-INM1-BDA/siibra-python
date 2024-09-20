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

from ..commons.logger import logger
from ..attributes.locations import Location
from ..attributes.attribute_collection import (
    AttributeCollection,
    MATRIX_INDEX_ENTITY_KEY,
    attr_of_general_interest,
)

SUMMARY_NAME = "Summary Tabular Data"


class Feature(AttributeCollection):
    schema: str = "siibra/concepts/feature/v0.2"

    def __str__(self):
        return self.__repr__()

    def __repr__(self) -> str:
        return f"Feature<name={self.name}>"

    @property
    def name(self):
        try:
            return super().name
        except Exception:
            # TODO: reconsider how to name unnamed features
            return f"Unnamed feature: {', '.join(map(str, self.modalities))}"

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
        from ..attributes.dataproviders import TabularDataProvider

        matrix_entity_key = self.filter(attr_of_general_interest)

        dfs: List[pd.DataFrame] = [
            d.get_data() for d in self._find(TabularDataProvider)
        ]
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

    def plotiter(self, *args, **kwargs):
        from ..attributes.dataproviders import TabularDataProvider

        for d in self._find(TabularDataProvider):
            yield d.plot(*args, **kwargs)

    def plot(self, *args, **kwargs):
        from ..attributes.dataproviders import TabularDataProvider

        summaries = [
            tdp for tdp in self._find(TabularDataProvider) if tdp.name == SUMMARY_NAME
        ]
        if len(summaries) > 0:
            return [dp.plot(*args, **kwargs) for dp in summaries]
        counter = 0
        return_val = []
        for plot in self.plotiter(*args, **kwargs):
            return_val.append(plot)
            counter += 1
            if counter > 5:
                logger.warning(
                    "Only plotting 5 plots. Please use plotiter if you intend to plot all plots"
                )
                break
        return return_val
