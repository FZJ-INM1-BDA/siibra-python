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
"""Non-preconfigured data features hosted at EBRAINS."""

from zipfile import ZipFile
from .. import anchor as _anchor
from .. import feature

from ...retrieval import datasets


DOI_TMPL = """
doi
---
{doi}
"""


class EbrainsDataFeature(feature.Feature, category="other"):
    def __init__(
        self,
        dataset_version_id: str,
        anchor: _anchor.AnatomicalAnchor
    ):
        feature.Feature.__init__(
            self,
            modality=None,  # lazy implementation below
            description=None,  # lazy implementation below
            anchor=anchor,
            datasets=[datasets.EbrainsV3DatasetVersion(id=dataset_version_id)],
        )
        self.version = None
        self._next = None
        self._prev = None

    @property
    def id(self):
        return self._dataset.id

    @property
    def _dataset(self) -> datasets.EbrainsV3DatasetVersion:
        assert len(self.datasets) == 1
        return self.datasets[0]

    @property
    def description(self) -> str:
        return self._dataset.description

    @property
    def name(self):
        if self._dataset.name.startswith(" "):
            return f"Ebrains Dataset: {self._dataset.is_version_of[0].name}"
        else:
            return f"Ebrains Dataset: {self._dataset.name}"

    @property
    def version_identifier(self):
        return self._dataset.version_identifier

    @property
    def version_history(self):
        return self._dataset.version_changes

    @property
    def url(self):
        return self._dataset.ebrains_page

    def __hash__(self):
        return hash(self._dataset)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, EbrainsDataFeature):
            return False
        return self._dataset == o._dataset

    def _export(self, fh: ZipFile):
        super()._export(fh)
        fh.writestr("doi.md", DOI_TMPL.format(doi=self.url))
