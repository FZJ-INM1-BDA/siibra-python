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

from . import regional_connectivity
from hashlib import md5


class FunctionalConnectivity(
    regional_connectivity.RegionalConnectivity,
    configuration_folder="features/connectivity/regional/functional",
    category="connectivity"
):
    """Functional connectivity matrix grouped by a parcellation."""

    def __init__(self, paradigm: str, **kwargs):
        regional_connectivity.RegionalConnectivity.__init__(self, **kwargs)
        self.paradigm = paradigm

        # paradign is used to distinguish functional connectivity features from each other.
        assert self.paradigm, "Functional connectivity must have paradigm defined!"

    @property
    def id(self):
        return super().id + "--" + md5(self.paradigm.encode("utf-8")).hexdigest()
