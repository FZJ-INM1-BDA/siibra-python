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

__all__ = ["cellular", "molecular", "fibres", "connectivity", "external"]

from . import (
    cellular,
    molecular,
    fibres,
    connectivity,
    external
)

from .basetypes.feature import Feature
from .basetypes.cortical_profile import CorticalProfile
from .basetypes.volume_of_interest import VolumeOfInterest
get = Feature.match


TYPES = Feature._get_subclasses()


def __dir__():
    return [
        "cellular",
        "molecular",
        "fibres",
        "connectivity",
        "external",
        "Feature",
        "get",
        "CorticalProfile",
        "VolumeOfInterest"
    ]
