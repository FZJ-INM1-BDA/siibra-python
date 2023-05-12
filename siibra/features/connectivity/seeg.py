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

class SEEG(
    regional_connectivity.RegionalConnectivity,
    configuration_folder="features/connectivity/regional/seeg",
    category="connectivity"
):
    """
    Connectivity matrix obtained in a semi-quantitative manner and grouped by a
    parcellation.
    """

    def __init__(self, **kwargs):
        self.paradigm = kwargs.pop("paradigm")
        regional_connectivity.RegionalConnectivity.__init__(self, **kwargs)

    @property
    def id(self):
        return super().id + "--" + md5(f"{self.modality}-{self.paradigm}".encode("utf-8")).hexdigest()
    
    @property
    def name(self):
        """Returns a short human-readable name of this feature."""
        spec = f' - {self.paradigm}' if len(self.paradigm)>0 else ''
        return f"{self.__class__.__name__} ({self.modality + spec}) anchored at {self.anchor}"