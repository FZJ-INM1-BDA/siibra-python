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
"""Metadata connection to EBRAINS datasets."""

from .requests import MultiSourcedRequest, GitlabProxy, GitlabProxyEnum

import re
from typing import Union, List
from abc import ABC, abstractproperty
from hashlib import md5

try:
    from typing import TypedDict
except ImportError:
    # support python 3.7
    from typing_extensions import TypedDict


class EbrainsDatasetUrl(TypedDict):
    url: str


EbrainsDatasetPerson = TypedDict("EbrainsDatasetPerson", {
    "@id": str,
    "schema.org/shortName": str,
    "identifier": str,
    "shortName": str,
    "name": str,
})

EbrainsDatasetEmbargoStatus = TypedDict("EbrainsDatasetEmbargoStatus", {
    "@id": str,
    "name": str,
    "identifier": List[str]
})


class EbrainsBaseDataset(ABC):
    @abstractproperty
    def id(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def name(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def urls(self) -> List[EbrainsDatasetUrl]:
        raise NotImplementedError

    @abstractproperty
    def description(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def contributors(self) -> List[EbrainsDatasetPerson]:
        raise NotImplementedError

    @abstractproperty
    def ebrains_page(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def custodians(self) -> List[EbrainsDatasetPerson]:
        raise NotImplementedError

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, o: object) -> bool:
        return hasattr(o, "id") and self.id == o.id

    def match(self, spec: Union[str, 'EbrainsBaseDataset']) -> bool:
        """
        Checks if the given specification describes this dataset.

        Parameters
        ----------
        spec (str, EbrainsBaseDataset)
            specification to be matched.
        Returns
        -------
        bool
        """
        if spec is self:
            return True
        if isinstance(spec, str):
            return self.id == spec
        raise RuntimeError(
            f"Cannot match {spec.__class__}, must be either str or EbrainsBaseDataset"
        )

    @property
    def LICENSE(self) -> str:
        license_ = self._detail.get("license", [])
        if len(license_) > 0:
            return license_ if isinstance(license_, str) else '\n'.join(license_)
        return None


class EbrainsDataset(EbrainsBaseDataset):
    """Ebrains dataset v1 connection"""

    def __init__(self, id, name=None, embargo_status: List[EbrainsDatasetEmbargoStatus] = None, *, cached_data=None):
        super().__init__()

        self._id = id
        self._name = name
        self._cached_data = cached_data
        self.embargo_status = embargo_status

        if id is None:
            raise TypeError("Dataset id is required")

        match = re.search(r"([a-f0-9-]+)$", id)
        if not match:
            raise ValueError(
                f"{self.__class__.__name__} initialized with invalid id: {self.id}"
            )

    @property
    def id(self) -> str:
        return self._id

    @property
    def _detail(self):
        if not self._cached_data:
            match = re.search(r"([a-f0-9-]+)$", self.id)
            instance_id = match.group(1)
            self._cached_data = MultiSourcedRequest(
                requests=[
                    GitlabProxy(
                        GitlabProxyEnum.DATASET_V1,
                        instance_id=instance_id,
                    ),
                ]
            ).data
        return self._cached_data

    @property
    def name(self) -> str:
        if self._name is None:
            self._name = self._detail.get("name")
        return self._name

    @property
    def urls(self) -> List[EbrainsDatasetUrl]:
        return [
            {
                "url": f if f.startswith("http") else f"https://doi.org/{f}",
            }
            for f in self._detail.get("kgReference", [])
        ]

    @property
    def description(self) -> str:
        return self._detail.get("description")

    @property
    def contributors(self) -> List[EbrainsDatasetPerson]:
        return self._detail.get("contributors")

    @property
    def ebrains_page(self):
        return f"https://search.kg.ebrains.eu/instances/{self.id}"

    @property
    def custodians(self) -> EbrainsDatasetPerson:
        return self._detail.get("custodians")


class EbrainsV3DatasetVersion(EbrainsBaseDataset):
    @staticmethod
    def _parse_person(d: dict) -> EbrainsDatasetPerson:
        assert "https://openminds.ebrains.eu/core/Person" in d.get("type"), "Cannot convert a non person to a person dict!"
        _id = d.get("id")
        name = f"{d.get('givenName')} {d.get('familyName')}"
        return {
            '@id': _id,
            'schema.org/shortName': name,
            'identifier': _id,
            'shortName': name,
            'name': name
        }

    def __init__(self, id, *, cached_data=None) -> None:
        super().__init__()

        self._id = id
        self._cached_data = cached_data

    @property
    def _detail(self):
        if not self._cached_data:
            match = re.search(r"([a-f0-9-]+)$", self._id)
            instance_id = match.group(1)
            self._cached_data = MultiSourcedRequest(
                requests=[
                    GitlabProxy(
                        GitlabProxyEnum.DATASETVERSION_V3,
                        instance_id=instance_id,
                    ),
                ]
            ).data
        return self._cached_data

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        fullname = self._detail.get("fullName")
        if not fullname:
            for dataset in self.is_version_of:
                if fullname:
                    break
                fullname = dataset.name
        version_id = self._detail.get("versionIdentifier")
        return f"{fullname} ({version_id})"

    @property
    def is_version_of(self):
        if not hasattr(self, "_is_version_of"):
            self._is_version_of = [EbrainsV3Dataset(id=id.get("id")) for id in self._detail.get("isVersionOf", [])]
        return self._is_version_of

    @property
    def urls(self) -> List[EbrainsDatasetUrl]:
        return [{
            "url": doi.get("identifier", None)
        } for doi in self._detail.get("doi", [])]

    @property
    def description(self) -> str:
        description = self._detail.get("description")
        for ds in self.is_version_of:
            if description:
                break
            description = ds.description
        return description or ""

    @property
    def contributors(self) -> List[EbrainsDatasetPerson]:
        return [EbrainsV3DatasetVersion._parse_person(d) for d in self._detail.get("author", [])]

    @property
    def ebrains_page(self) -> str:
        if len(self.urls) > 0:
            return self.urls[0].get("url")
        return None

    @property
    def custodians(self) -> EbrainsDatasetPerson:
        return [EbrainsV3DatasetVersion._parse_person(d) for d in self._detail.get("custodian", [])]

    @property
    def version_changelog(self):
        return self._detail.get("versionInnovation", "")

    @property
    def version_identifier(self):
        return self._detail.get("versionIdentifier", "")


class EbrainsV3Dataset(EbrainsBaseDataset):
    def __init__(self, id, *, cached_data=None) -> None:
        super().__init__()

        self._id = id
        self._cached_data = cached_data
        self._contributers = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._detail.get("fullName")

    @property
    def urls(self) -> List[EbrainsDatasetUrl]:
        return [{
            "url": doi.get("identifier", None)
        } for doi in self._detail.get("doi", [])]

    @property
    def _detail(self):
        if not self._cached_data:
            match = re.search(r"([a-f0-9-]+)$", self._id)
            instance_id = match.group(1)
            self._cached_data = MultiSourcedRequest(
                requests=[
                    GitlabProxy(
                        GitlabProxyEnum.DATASET_V3,
                        instance_id=instance_id,
                    ),
                ]
            ).data
        return self._cached_data

    @property
    def description(self) -> str:
        return self._detail.get("description", "")

    @property
    def contributors(self):
        if self._contributers is None:
            contributers = {}
            for version_id in self.version_ids:
                contributers.update(
                    {c['@id']: c for c in EbrainsV3DatasetVersion(version_id).contributors}
                )
            self._contributers = list(contributers.values())
        return self._contributers

    @property
    def ebrains_page(self) -> str:
        if len(self.urls) > 0:
            return self.urls[0].get("url")
        return None

    @property
    def custodians(self) -> EbrainsDatasetPerson:
        return [EbrainsV3DatasetVersion._parse_person(d) for d in self._detail.get("custodian", [])]

    @property
    def version_ids(self) -> List['str']:
        return [version.get("id") for version in self._detail.get("versions", [])]


class GenericDataset():

    def __init__(
        self,
        name: str = None,
        contributors: List[str] = None,
        url: str = None,
        description: str = None,
        license: str = None
    ):
        self._name = name
        self._contributors = contributors
        self._url = url
        self._description = description
        self._license = license

    @property
    def contributors(self):
        return [{"name": cont} for cont in self._contributors]

    @property
    def id(self) -> str:
        return md5(self.name.encode('utf-8')).hexdigest()

    @property
    def name(self) -> str:
        return self._name

    @property
    def LICENSE(self) -> str:
        return self._license

    @property
    def urls(self) -> List[EbrainsDatasetUrl]:
        return [{"url": self._url}]

    @property
    def description(self) -> str:
        return self._description

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, o: object) -> bool:
        return hasattr(o, "id") and self.id == o.id
