# Copyright 2018-2025
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
"""Multimodal data features in 2D section."""

from . import image


class CellbodyStainedSection(
    image.Image,
    configuration_folder='features/images/sections/cellbody',
    category="cellular"
):
    def __init__(self, **kwargs):
        image.Image.__init__(self, **kwargs, modality="cell body staining")
