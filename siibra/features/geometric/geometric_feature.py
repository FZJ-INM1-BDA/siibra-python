# Copyright 2018-2026
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
"""Naming and identity shared by features based on geometries."""

from hashlib import md5

from ..feature import Feature


class _GeometricFeature:
    """
    Mixin for classes deriving from both Feature and Geometry. It is not a
    Feature itself, so that it does not become a feature type to query.
    """

    @property
    def name(self) -> str:
        if not self._name:
            return Feature.name.fget(self)
        name = f"{self._name} ({self.modality})"
        return f"[PRERELEASE] {name}" if self._prerelease else name

    @property
    def id(self) -> str:
        # Feature.id hashes the anchor, whose text lists its regions in set
        # order. That order differs between sessions, so hash the data instead.
        if self._id:
            return self._id
        prefix = next((f"{ds.id}--" for ds in self.datasets if hasattr(ds, "id")), "")
        return prefix + md5(f"{self.name} - {self._identity}".encode("utf-8")).hexdigest()
