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

from .serializable_concept import JSONSerializable
from ..commons import logger
from ..retrieval import EbrainsKgQuery
from ..openminds.core.v4.products.datasetVersion import Model as DatasetVersionModel
from ..openminds.base import ConfigBaseModel
from ..registry import Preconfigure, REGISTRY

import hashlib
import re
from datetime import date
from typing import Any, Dict, List, Optional, Union
from pydantic import Field


class Url(ConfigBaseModel):
    doi: str
    cite: Optional[str]


class Dataset(JSONSerializable):
    """Parent class for datasets. Each dataset has an identifier."""

    REGISTRY = {}

    def __init__(self, identifier, description=""):
        self.id = identifier
        self._description_cached = description

    def __str__(self):
        return f"{self.__class__.__name__}: {self.id}"

    def __init_subclass__(cls, type_id=None):
        if type_id in Dataset.REGISTRY:
            logger.warning(
                f"Type id '{type_id}' already provided by {Dataset.REGISTRY[type_id].__name__}, but {cls.__name__} suggests itself as well"
            )
        if type_id is not None:
            logger.debug(f"Registering specialist {cls.__name__} for type id {type_id}")
            Dataset.REGISTRY[type_id] = cls
        cls.type_id = type_id
        return super().__init_subclass__()

    @property
    def is_volume(self):
        """Return True if this dataset represents a brain volume source.

        This property is overwritten by siibra.volumes.VolumeSrc
        """
        return False

    @property
    def is_surface(self):
        """Return True if this dataset represents a brain volume surface.

        This property is overwritten by siibra.volumes.VolumeSrc
        """
        return False

    @property
    def publications(self):
        """
        List of publications for this dataset.
        Empty list here, but implemented in some derived classes.
        """
        return []

    @property
    def urls(self):
        """
        List of URLs related to this dataset.
        Empty list here, but implemented in some derived classes.
        """
        return []

    @property
    def description(self):
        """
        Textual description of Dataset.
        Empty string here, but implemented in some derived classes.
        """
        return self._description_cached

    @classmethod
    def extract_type_id(cls, spec):
        for key in ["@type", "kgSchema"]:
            if key in spec:
                return spec[key]
        raise RuntimeError(f"No type defined in dataset specification: {spec}")

    @classmethod
    def get_model_type(Cls):
        return "https://openminds.ebrains.eu/core/DatasetVersion"

    @property
    def model_id(self):
        _id = hashlib.md5(
            str(self.id if self.id else f"{str(self)}{self.description}").encode(
                "utf-8"
            )
        ).hexdigest()
        return f"{self.get_model_type()}/{_id}"

    def to_model(self, **kwargs) -> "DatasetJsonModel":
        metadata = DatasetVersionModel(
            id=self.model_id,
            type=self.get_model_type(),
            accessibility={"@id": self.embargo_status[0].get("@id")}
            if hasattr(self, "embargo_status")
            and self.embargo_status is not None
            and len(self.embargo_status) == 1
            else {
                "@id": "https://openminds.ebrains.eu/instances/productAccessibility/freeAccess"
            },
            data_type=[
                {
                    "@id": "https://openminds.ebrains.eu/instances/semanticDataType/derivedData"
                }
            ],
            digital_identifier={"@id": None},
            ethics_assessment={"@id": None},
            experimental_approach=[{"@id": None}],
            full_documentation={"@id": None},
            full_name=self.name if hasattr(self, "name") else None,
            license={"@id": None},
            release_date=date(1970, 1, 1),
            short_name=self.name[:30] if hasattr(self, "name") else "",
            technique=[{"@id": None}],
            version_identifier="",
            version_innovation="",
            description=(self.description or "")[:2000]
            if hasattr(self, "description")
            else "",
        )
        return DatasetJsonModel(
            id=metadata.id,
            type=Dataset.get_model_type(),
            metadata=metadata,
            urls=[Url(**url) for url in self.urls],
        )


class DatasetJsonModel(ConfigBaseModel):
    id: str = Field(..., alias="@id")
    type: str = Field(Dataset.get_model_type(), alias="@type", const=True)
    metadata: DatasetVersionModel
    urls: List[Url]


class OriginDescription(Dataset, type_id="fzj/tmp/simpleOriginInfo/v0.0.1"):
    def __init__(self, name, description, urls):
        Dataset.__init__(self, None, description=description)
        # we model the following as property functions,
        # so derived classes may replace them with a lazy loading mechanism.
        self.name = name
        self._urls = urls

    @property
    def urls(self):
        return self._urls

    @classmethod
    def _from_json(cls, spec):
        type_id = cls.extract_type_id(spec)
        assert type_id == cls.type_id
        return cls(
            name=spec["name"],
            description=spec.get("description"),
            urls=spec.get("url", []),
        )
    
@Preconfigure("snapshots/ebrainsquery/v1")
class EbrainsDataset(Dataset, type_id="minds/core/dataset/v1.0.0"):
    def __init__(self, id, name, embargo_status=None, *, cached_data=None, **kwargs):
        Dataset.__init__(self, id, description=None)
        self._cached_data = cached_data
        self.embargo_status = embargo_status
        self._name_cached = name
        
        if id is None:
            raise TypeError("Dataset id is required")

        match = re.search(r"([a-f0-9-]+)$", id)
        if not match:
            raise ValueError(
                f"{self.__class__.__name__} initialized with invalid id: {self.id}"
            )

    @property
    def detail(self):
        if not self._cached_data:
            match = re.search(r"([a-f0-9-]+)$", self.id)
            self._cached_data = EbrainsKgQuery(
                query_id="interactiveViewerKgQuery-v1_0",
                instance_id=match.group(1),
                params={"vocab": "https://schema.hbp.eu/myQuery/"},
            ).data
        return self._cached_data

    @property
    def name(self):
        if self._name_cached is None:
            self._name_cached = self.detail.get("name")
        return self._name_cached

    @property
    def publications(self):
        return self.detail.get("publications")

    @property
    def urls(self):
        return [
            {
                "doi": f,
            }
            for f in self.detail.get("kgReference", [])
        ]

    @property
    def description(self):
        return self.detail.get("description")

    @property
    def contributors(self):
        return self.detail.get("contributors")

    @property
    def ebrains_page(self):
        return f"https://search.kg.ebrains.eu/instances/{self.id}"

    @property
    def custodians(self):
        return self.detail.get("custodians")

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, o: object) -> bool:
        if type(o) is not EbrainsDataset and not issubclass(type(o), EbrainsDataset):
            return False
        return self.id == o.id

    @classmethod
    def _from_json(cls, spec):
        """the _from_json method from EbrainsDataset can be called in two instances:
        1/ when core concepts are being initialized
        2/ when boostraped from configuration
        """

        only_id_flag = False
        # when constructed from core concepts, directly fetch from registry
        if all(key in spec for key in ("@type", "kgSchema", "kgId")):
            only_id_flag = True
            try:
                return REGISTRY[EbrainsDataset][spec.get("kgId")]
            except:
                pass
        
        # otherwise, construct the instance
        found_id = re.search(r'[a-f0-9-]+$', spec.get("fullId") or spec.get("kgId"))
        assert found_id, f"Expecting spec.fullId or spec.kgId to match '[a-f0-9-]+$', but did not."
        return cls(
            id=found_id.group(),
            name=spec.get("name"),
            embargo_status=spec.get("embargo_status", None),
            cached_data=spec if not only_id_flag else None,
        )

    def match(self, spec: Union[str, 'EbrainsDataset']) -> bool:
        """Checks of a given spec (of type str or EbrainsDataset) describes this dataset.

        Args:
            spec (str, EbrainsDataset): spec to be checked
        """
        if spec is self:
            return True
        if isinstance(spec, str):
            return self.id == spec
        raise RuntimeError(f"Cannot match {spec.__class__}, must be either str or EbrainsDataset")
