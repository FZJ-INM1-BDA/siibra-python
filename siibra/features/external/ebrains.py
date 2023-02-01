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

# simple data features anchored to a point in space

from .. import anchor as _anchor
from ..basetypes import feature

from ...retrieval import datasets


class EbrainsDataFeature(feature.Feature, datasets.EbrainsDataset):

    def __init__(
        self,
        dataset_id: str,
        name: str,
        anchor: _anchor.AnatomicalAnchor,
        embargo_status: str = None,
    ):
        feature.Feature.__init__(
            self,
            modality=None,  # lazy implementation below
            description=None,  # lazy implementation below
            anchor=anchor,
            datasets=[]
        )
        datasets.EbrainsDataset.__init__(
            self,
            id=dataset_id,
            name=name,
            embargo_status=embargo_status,
        )
        self.version = None
        self._next = None
        self._prev = None

    @property
    def id(self):
        # There exists a property name collision (id property implemented by both Feature and dataset.EbrainsDataset)
        # Explicitly use datasets.EbrainsDataset's implementation of id
        # We could fix this by reordering the mro, but I feel the below implementation is more explicit.
        return datasets.EbrainsDataset.id.fget(self)

    @property
    def modality(self):
        return ", ".join(self.detail.get('methods', []))

    @property
    def description(self):
        return self.detail.get("description", "")

    @property
    def name(self):
        return self._name_cached

    @property
    def version_history(self):
        if self._prev is None:
            return [self.version]
        else:
            return [self.version] + self._prev.version_history

    @property
    def url(self):
        return f"https://search.kg.ebrains.eu/instances/{self.id.split('/')[-1]}"

    def __str__(self):
        return datasets.EbrainsDataset.__str__(self)

    def __hash__(self):
        return datasets.EbrainsDataset.__hash__(self)

    def __eq__(self, o: object) -> bool:
        return datasets.EbrainsDataset.__eq__(self, o)
