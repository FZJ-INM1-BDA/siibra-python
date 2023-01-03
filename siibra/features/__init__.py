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

from ._basetypes.feature import Feature as _Feature
from ._basetypes.volume_of_interest import VolumeOfInterest
from ._basetypes.cortical_profile import CorticalProfile as _CorticalProfile
get = _Feature.match


ALL = _Feature._get_visible_subclass_names()
PROFILES = _CorticalProfile._get_visible_subclass_names()
