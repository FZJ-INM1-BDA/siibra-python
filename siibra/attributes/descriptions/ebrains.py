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

from dataclasses import dataclass
from typing import Dict, List, Union, TYPE_CHECKING
import json

from .base import Description
from .doi import Doi
from ...operations.file_fetcher.dataproxy_fetcher import DataproxyRepository
from ...commons.string import extract_uuid

if TYPE_CHECKING:
    from ...factory.livequery.ebrains import EbrainsQuery


@dataclass
class EbrainsRef(Description):
    schema = "siibra/attr/desc/ebrains/v0.1"
    ids: Dict[str, Union[str, List[str]]] = None

    @property
    def _dataset_verion_ids(self) -> List[str]:
        from ...factory.livequery.ebrains import EbrainsQuery

        return [
            extract_uuid(id)
            for key, value in self.ids.items()
            if key == "openminds/DatasetVersion"
            for id in EbrainsQuery.iter_ids(value)
        ]

    @property
    def dois(self) -> List[Doi]:
        result = []
        for dsv in self._dataset_verion_ids:
            dsv_obj = EbrainsQuery.get_dsv(dsv)
            for doi in dsv_obj["doi"]:
                result.append(Doi(value=doi["identifier"]))
        return result

    @property
    def descriptions(self) -> List[str]:
        from ...factory.livequery.ebrains import EbrainsQuery

        result: List[str] = []

        for v in self._dataset_verion_ids:
            j = EbrainsQuery.get_dsv(v)
            pev_desc = j.get("description")
            if pev_desc:
                result.append(pev_desc)
            for pe_ref in j.get("isVersionOf", []):
                extracted_id = extract_uuid(pe_ref)
                pe = EbrainsQuery.get_ds(extracted_id)
                pe_desc = pe.get("description")
                if pe_desc:
                    result.append(pe_desc)
        return result
