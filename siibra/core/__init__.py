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

from siibra.commons import TypedRegistry
from .concept import AtlasConcept
from .atlas import Atlas
from .space import Space, Point, PointSet, BoundingBox
from .region import Region
from .parcellation import Parcellation
from .datasets import Dataset, OriginDescription, EbrainsDataset

# initialize the core concepts and their bootstrapped registries
spaces: TypedRegistry[Space] = Space.REGISTRY
parcellations: TypedRegistry[Parcellation] = Parcellation.REGISTRY
atlases: TypedRegistry[Atlas] = Atlas.REGISTRY
