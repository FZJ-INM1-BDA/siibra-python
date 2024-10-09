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
from typing import List, Any, Dict, Iterable, Tuple, BinaryIO, Union, ClassVar
import pandas as pd
from ..commons.logger import logger


def key_is_extra(key: str):
    return key.startswith("x-")


@dataclass
class Attribute:
    """Base clase for attributes."""

    IGNORE_KEYS: ClassVar = {"schema", "annotates", "extra", "id", "key"}
    SCHEMAS: ClassVar = {}

    schema: str = field(default="siibra/attr", init=False, repr=False)
    name: str = field(default=None, repr=False)

    # TODO to be deprecated
    id: str = field(default=None, repr=False)
    annotates: str = field(default=None, repr=False)

    # TODO performance implications? may have to set hash=False
    extra: Dict[str, Any] = field(default_factory=dict, repr=False, hash=False)

    # derived classes set their schema as class parameter
    # to skip automatic json decoding (e.g. adhoc dataclasses & extended class) set schema to None
    def __init_subclass__(cls):
        if cls.schema is None:
            return
        assert (
            cls.schema != Attribute.schema
        ), "Subclassed attributes must have unique schemas"
        assert cls.schema not in cls.SCHEMAS, f"{cls.schema} already registered."
        cls.SCHEMAS[cls.schema] = cls

    @staticmethod
    def from_dict(json_dict: Dict[str, Any]) -> List["Attribute"]:
        """Generating a list of attributes from a dictionary.
        TODO consider moving this to siibra.factory.factory and have a single build_object call
        """

        att_type: str = json_dict.pop("@type")
        if att_type.startswith("x-"):
            attr = Attribute(extra=json_dict)
            attr.schema = att_type
            return [attr]
        Cls = Attribute.SCHEMAS.get(att_type)
        if Cls is None:
            logger.warning(f"Cannot parse type {att_type}")
            return []

        _ = json_dict.pop("mapping", None)
        try:
            return_attr: "Attribute" = Cls(
                **{key: json_dict[key] for key in json_dict if not key_is_extra(key)}
            )
        except TypeError as e:
            print("Error Attribute.from_dict:", json_dict, e)
            raise e
        for key in json_dict:
            if key_is_extra(key):
                return_attr.extra[key] = json_dict[key]
        return [return_attr]

    def _iter_zippable(
        self,
    ) -> Iterable[Tuple[str, Union[str, None], Union[BinaryIO, None]]]:
        """
        This method allows attributes to expose what kind of data to be written when user calls `to_zip`.
        Attribute collection will iterate over all _iter_zippable. For each:

        - append the TextDesc to the main desc file
        - (if provided) iterate over all bytes from binary io, and write to ...
        - filename(.suffix, if provided)

        This method returns an iterable, as multiple files *may* need to be written.

        If binaryio not provided, subclass hints that no file needs to be written.

        Yields:
            Tuple[str, str, BinaryIO]: Tuple[TextDesc, suffix (includes leading dot) or none, binaryio or none]
        """
        return []
