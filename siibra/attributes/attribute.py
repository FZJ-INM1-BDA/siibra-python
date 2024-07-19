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

from dataclasses import dataclass, field
from typing import List, Any, Dict
from ..commons_new.logger import logger

SCHEMAS = {}


def key_is_extra(key: str):
    return key.startswith("x-") or key.startswith("facet/")


@dataclass
class Attribute:
    """Base clase for attributes."""

    schema: str = field(default="siibra/attr", init=False, repr=False)
    id: str = field(default=None, repr=False)
    annotates: str = field(default=None, repr=False)

    # TODO performance implications? may have to set hash=False
    extra: Dict[str, Any] = field(default_factory=dict, repr=False, hash=False)

    # derived classes set their schema as class parameter
    def __init_subclass__(cls):
        assert (
            cls.schema != Attribute.schema
        ), "Subclassed attributes must have unique schemas"
        assert cls.schema not in SCHEMAS, f"{cls.schema} already registered."
        SCHEMAS[cls.schema] = cls

    @staticmethod
    def from_dict(json_dict: Dict[str, Any]) -> List["Attribute"]:
        """Generating a list of attributes from a dictionary.
        TODO consider moving this to siibra.factory.factory and have a single build_object call
        """

        att_type: str = json_dict.pop("@type")
        if att_type.startswith("x-"):
            return []
        Cls = SCHEMAS.get(att_type)
        if Cls is None:
            logger.warning(f"Cannot parse type {att_type}")
            return []

        return_attr: "Attribute" = Cls(
            **{key: json_dict[key] for key in json_dict if not key_is_extra(key)}
        )
        for key in json_dict:
            if key_is_extra(key):
                return_attr.extra[key] = json_dict[key]
        return [return_attr]

    @property
    def facets(self):
        from ..attributes.descriptions import Facet

        # TODO use str.removeprefix when py3.9 is the lowest python version supported
        return [
            Facet(key=key.replace("facet/", ""), value=self.extra[key])
            for key in self.extra
            if key.startswith("facet/")
        ]