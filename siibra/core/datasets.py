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

from datetime import datetime
from typing import Any, Dict, List
from ..commons import logger
from ..retrieval import EbrainsRequest
from ..openminds.core.v4.products import datasetVersion
import re


class Dataset:
    """Parent class for datasets. Each dataset has an identifier."""

    REGISTRY = {}
    _id = None

    def __init__(self, identifier, description=""):
        self._id = identifier
        self._description_cached = description

    def __str__(self):
        return f"{self.__class__.__name__}: {self._id}"

    def __init_subclass__(cls, type_id=None):
        if type_id in Dataset.REGISTRY:
            logger.warning(
                f"Type id '{type_id}' already provided by {Dataset.REGISTRY[type_id].__name__}, but {cls.__name__} suggests itself as well"
            )
        if type_id is not None:
            logger.debug(f"Registering specialist {cls.__name__} for type id {type_id}")
            Dataset.REGISTRY[type_id] = cls
        cls.type_id = type_id

    @property
    def is_image_volume(self):
        """Overwritten by derived dataset classes in the siibra.volumes"""
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


class EbrainsDataset(datasetVersion.Model):
    """
    n.b. the lazy laoding of KG dataset means that datasetVersion.Model.__init__ is called lazily.
    (when accessing .detail attr). When it is called, it wipes all existing attr.
    """
    _id = None
    _detail = None
    _detail_loader = None
    _name_cached = None
    def __init__(self, id):
        self._detail = None
        if id is None:
            raise TypeError("Dataset id is required")
        
        self._id = id

        match = re.search(r"([a-f0-9-]+)$", id)
        if not match:
            raise ValueError(
                f"{self.__class__.__name__} initialized with invalid id: {self.id}"
            )
        self._detail_loader = EbrainsRequest(
            query_id="interactiveViewerKgQuery-v1_0",
            instance_id=match.group(1),
            params={"vocab": "https://schema.hbp.eu/myQuery/"},
        )

    def dict(self, *arg, **kwarg):
        """
        overriding Model.dict method
        accessing self.detail will lazily load the KG data.
        
        """
        self._lazy_load()
        return datasetVersion.Model.dict(self, *arg, **kwarg)

    def __getattr__(self, attr: str):
        """
        getattr will only be called if attr is not defined the normal way
        this might be a sign that the dataset hasn't yet been lazy loaded.
        """
        self._lazy_load()
        return getattr(self, attr)

    def _lazy_load(self):
        if self._detail:
            return
        
        _detail = self._detail_loader.data
        _dataset = EbrainsDataset.parse_legacy_kg(_detail)
        datasetVersion.Model.__init__(self, **_dataset.dict())
        self._detail = _detail

    @property
    def detail(self):
        """
        Lazy loading of dataset detail retrieved from Ebrains KG.
        It will also call datasetVersion.Model.__init__ if it has not yet.
        This will validate the detail and populate the attributes.
        """
        self._lazy_load()
        return self._detail

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
    def custodians(self):
        return self.detail.get("custodians")

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, o: object) -> bool:
        if type(o) is not EbrainsDataset and not issubclass(type(o), EbrainsDataset):
            return False
        return self._id == o._id

    @staticmethod
    def parse_legacy_id(legacy_id: str) -> str:
        return legacy_id

    @staticmethod
    def parse_legacy_kg(kg_result: Dict[str, Any]) -> datasetVersion.Model:
        """
        converts legacy kg result to datasetVersion instance
        (pass lazily loaded legacy kg result to this method to get openminds (current) datasetVersionModel)
        """

        dois: List[str] = kg_result.get("kgReference", [])
        if len(dois) == 0:
            logger.warning(f"kgReference length == 0, use fallback doi")
            dois=["UNKNOWN"]
        if len(dois) > 1:
            logger.warning(f"kgReference length > 1, has length: {len(dois)}, only using first one.")
        digital_identifier={
            "@id": dois[0],
        }

        licenses: List[Dict[str, str]] = kg_result.get("licenseInfo", [])
        if len(licenses) == 0:
            logger.warning(f"kgReference length == 0, use fallback doi")
            licenses=[{
                "name": "unknown license",
                "@id": "UNKNOWN",
            }]
        if len(licenses) > 1:
            logger.warning(f"kgReference length > 1, has length: {len(licenses)}, only using first one.")
        license=licenses[0]

        release_date = datetime(1970, 1, 1)
        
        return datasetVersion.Model(
            id=EbrainsDataset.parse_legacy_id(kg_result.get("fullId")),
            type="https://openminds.ebrains.eu/core/DatasetVersion",
            accessibility={
                "@id": "https://openminds.ebrains.eu/instances/productAccessibility/freeAccess"
            },
            author=kg_result.get("contributors"),
            custodian=kg_result.get("custodians"),
            data_type=[{
                "@id": "https://openminds.ebrains.eu/instances/semanticDataType/experimentalData"
            }],
            description=kg_result.get("description")[:2000],
            digital_identifier=digital_identifier,

            # ethnic assessment cannot be directly parsed
            ethics_assessment={
                "@id": "https://openminds.ebrains.eu/instances/ethicsAssessment/unknown"
            },
            experimental_approach=[{
                "@id": "https://openminds.ebrains.eu/instance/ExperimentalApproach/unknown"
            }],
            full_documentation=digital_identifier,
            full_name=kg_result.get("name"),
            license=license,
            release_date=release_date,
            short_name=kg_result.get("name")[:30],
            technique=[{
                "@id": "unknown technique"
            }],
            version_identifier=kg_result.get("id"),
            version_innovation="Placeholder Innovation"
        )

    @classmethod
    def parse_legacy(Cls, json_input: Dict[str, Any]) -> 'EbrainsDataset':
        """
        parse_legacy may pass only the id (and optionally the schema)
        detail will only be filled lazily
        """
        id = json_input.get("kgId") or json_input.get("id")
        return Cls(id=id)
