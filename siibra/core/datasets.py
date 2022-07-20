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
from ..retrieval.requests import DECODERS, EbrainsRequest
from ..openminds.core.v4.products.datasetVersion import Model as DatasetVersionModel
from ..openminds.base import ConfigBaseModel

import hashlib
import re
from datetime import date
from typing import Any, Dict, List, Optional
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
            str(
                self.id if self.id else f"{str(self)}{self.description}"
            ).encode("utf-8")
        ).hexdigest()
        return f'{self.get_model_type()}/{_id}'

    def to_model(self, **kwargs) -> 'DatasetJsonModel':
        metadata=DatasetVersionModel(
            id=self.model_id,
            type=self.get_model_type(),
            accessibility={ "@id": self.embargo_status[0].get("@id") } \
                if hasattr(self, 'embargo_status') \
                and self.embargo_status is not None \
                and len(self.embargo_status) == 1 \
                else { "@id": "https://openminds.ebrains.eu/instances/productAccessibility/freeAccess" },
            data_type=[{
                "@id": "https://openminds.ebrains.eu/instances/semanticDataType/derivedData"
            }],
            digital_identifier={
                "@id": None
            },
            ethics_assessment={
                "@id": None
            },
            experimental_approach=[{
                "@id": None
            }],
            full_documentation={
                "@id": None
            },
            full_name=self.name if hasattr(self, "name") else None,
            license={
                "@id": None
            },
            release_date=date(1970,1,1),
            short_name=self.name[:30] if hasattr(self, "name") else "",
            technique=[{
                "@id": None
            }],
            version_identifier="",
            version_innovation="",
            description=(self.description or "")[:2000] if hasattr(self, "description") else "",
        )
        return DatasetJsonModel(
            id=metadata.id,
            type=Dataset.get_model_type(),
            metadata=metadata,
            urls=[Url(**url) for url in self.urls]
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


class _EbrainsKgV3Base:
    BASE_URL = "https://core.kg.ebrains.eu/v3-beta/queries"
    QUERY_ID = None
    STAGE = "RELEASED"

    def __init__(self, _id_spec: Dict[str, Any]) -> None:
        self._id_spec = _id_spec
        self._spec = None
        

    @classmethod
    def _query(cls, spec: Dict[str, Any]=None):
        
        # for easy compatibility with data returned by KG
        # KG no longer uses @id for key, but instead id (by default)
        # also id prepends domain information (e.g. https://kg.ebrains.eu/api/instances/{uuid})
        # therefore, also extract the uuid protion

        if spec is not None:
            at_id=spec.get("@id")
            kg_id = spec.get("id")
            kg_id_search = kg_id and re.search(r'[a-f0-9-]+$', kg_id)
        
            assert at_id is not None or kg_id_search is not None
            uuid = at_id or kg_id_search.group()

        assert hasattr(cls, 'type_id')
        
        # for easy compatibility with data returned by KG
        # KG no longer uses @type for key, but type
        # also type is List[str]
        assert spec.get("@type") == cls.type_id or any ([t == cls.type_id for t in spec.get("type", [])])
        
        
        url=f"{cls.BASE_URL}/{cls.QUERY_ID}/instances?stage={cls.STAGE}"
        if spec is not None:
            url += f"&instanceId={uuid}"

        result = EbrainsRequest(url, DECODERS['.json']).get()

        assert 'data' in result

        if spec is not None:
            assert result.get('total') == 1
            assert result.get('size') == 1
            return result.get("data")[0]

        return result.get('data', [])
        

class EbrainsKgV3Dataset(Dataset, _EbrainsKgV3Base, type_id="https://openminds.ebrains.eu/core/Dataset"):
    BASE_URL = "https://core.kg.ebrains.eu/v3-beta/queries"
    QUERY_ID = "138111f9-1aa4-43f5-8e0a-6e6ed085fa3e"

    def __init__(self, spec: Dict[str, Any]):
        super().__init__(None)
        found = re.search(r'[a-f0-9-]+$', spec.get('id'))
        assert found
        self.id = found.group()
        self._description_cached = spec.get("description")
        self._spec = spec

    @classmethod
    def _from_json(cls, spec: Dict[str, Any]):
        json_obj = cls._query(spec)
        return cls(json_obj)


class EbrainsKgV3DatasetVersion(Dataset, _EbrainsKgV3Base, type_id="https://openminds.ebrains.eu/core/DatasetVersion"):
    
    BASE_URL = "https://core.kg.ebrains.eu/v3-beta/queries"
    QUERY_ID = "f7489d01-2f90-410c-9812-9ee7d10cc5be"

    def __init__(self, _id_spec: Dict[str, Any]):
        _EbrainsKgV3Base.__init__(self, _id_spec)
        Dataset.__init__(self, None)

    @classmethod
    def _from_json(cls, spec: Dict[str, Any]):
        return cls(spec)
    
    @property
    def description(self):
        if self._spec is None:
            self._spec = self._query(self._id_spec)
        
        self._description_cached = self._spec.get("description")
        
        if self._description_cached is not None and self._description_cached != '':
            return self._description_cached

        parent_datasets = self._spec.get("belongsTo", [])
        if len(parent_datasets) == 0:
            return None
        if len(parent_datasets) > 1:
            logger.warn(f"EbrainsKgV3DatasetVersion.description: more than one parent dataset found. Using the first one...")

        parent = EbrainsKgV3Dataset._from_json(parent_datasets[0])
        return parent.description



class EbrainsDataset(Dataset, type_id="minds/core/dataset/v1.0.0"):
    def __init__(self, id, name, embargo_status=None):
        Dataset.__init__(self, id, description=None)
        self.embargo_status = embargo_status
        self._name_cached = name
        self._detail = None
        if id is None:
            raise TypeError("Dataset id is required")

        match = re.search(r"([a-f0-9-]+)$", id)
        if not match:
            raise ValueError(
                f"{self.__class__.__name__} initialized with invalid id: {self.id}"
            )
        self._detail_loader = EbrainsKgQuery(
            query_id="interactiveViewerKgQuery-v1_0",
            instance_id=match.group(1),
            params={"vocab": "https://schema.hbp.eu/myQuery/"},
        )

    @property
    def detail(self):
        return self._detail_loader.data

    @property
    def name(self):
        if self._name_cached is None:
            self._name_cached = self.detail.get("name")
        return self._name_cached

    @property
    def publications(self):
        return self.detail.get('publications')

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
        type_id = cls.extract_type_id(spec)
        assert type_id == cls.type_id
        return cls(
            id=spec.get("kgId"),
            name=spec.get("name"),
            embargo_status=spec.get("embargo_status", None),
        )
