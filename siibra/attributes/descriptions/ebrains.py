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
from typing import Dict, List, Union
import json

from .base import Description
from ...retrieval.file_fetcher.dataproxy_fetcher import DataproxyRepository
from ...commons_new.string import extract_uuid

@dataclass
class EbrainsRef(Description):
    schema = "siibra/attr/desc/ebrains/v0.1"
    ids: Dict[str, Union[str, List[str]]] = None

    @property
    def descriptions(self) -> List[str]:
        repo = DataproxyRepository("reference-atlas-data")
        result: List[str] = []
        for key, value in self.ids.items():
            if key != "openminds/DatasetVersion":
                continue

            if isinstance(value, str):
                value = [value]

            assert (
                isinstance(value, list)
                and all(isinstance(v, str) for v in value)
            ), f"Expected all ids to be str, but was not {value}"

            for v in value:
                j = json.loads(repo.get(f"ebrainsquery/v3/DatasetVersion/{v}.json"))
                pev_desc = j.get("description")
                if pev_desc:
                    result.append(pev_desc)
                for pe_ref in j.get("isVersionOf", []):
                    extracted_id = extract_uuid(pe_ref)
                    pe = json.loads(repo.get(f"ebrainsquery/v3/Dataset/{extracted_id}.json"))
                    pe_desc = pe.get("description")
                    if pe_desc:
                        result.append(pe_desc)
        return result
