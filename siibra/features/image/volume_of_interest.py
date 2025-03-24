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
"""Multimodal data features as volumes."""

from . import image


class CellBodyStainedVolumeOfInterest(
    image.Image,
    configuration_folder="features/images/vois/cellbody",
    category="cellular"
):
    def __init__(self, **kwargs):
        image.Image.__init__(self, **kwargs, modality="cell body staining")


class BlockfaceVolumeOfInterest(
    image.Image,
    configuration_folder="features/images/vois/blockface",
    category="macrostructural"
):
    def __init__(self, **kwargs):
        image.Image.__init__(self, **kwargs, modality="blockface")


class DTIVolumeOfInterest(
    image.Image,
    configuration_folder="features/images/vois/blockface",
    category="fibres"
):
    def __init__(self, modality, **kwargs):
        image.Image.__init__(self, **kwargs, modality=modality)


class PLIVolumeOfInterest(
    image.Image,
    configuration_folder="features/images/vois/pli",
    category="fibres"
):
    def __init__(self, modality, **kwargs):
        image.Image.__init__(self, **kwargs, modality=modality)


class MRIVolumeOfInterest(
    image.Image,
    configuration_folder="features/images/vois/mri",
    category="macrostructural"
):
    def __init__(self, modality, **kwargs):
        image.Image.__init__(self, **kwargs, modality=modality)


class XPCTVolumeOfInterest(
    image.Image,
    configuration_folder="features/images/vois/xpct",
    category="cellular"
):
    def __init__(self, modality, **kwargs):
        image.Image.__init__(self, **kwargs, modality=modality)


class LSFMVolumeOfInterest(
    image.Image,
    configuration_folder="features/images/vois/lsfm",
    category="cellular"
):
    def __init__(self, modality, **kwargs):
        image.Image.__init__(self, **kwargs, modality=modality)

class MorphometryVolumeOfInterest(
    image.Image,
    configuration_folder="features/images/vois/morphometry",
    category="macrostructural"
):
    def __init__(self, modality, **kwargs):
        image.Image.__init__(self, **kwargs, modality=modality)

# class SegmentedVolumeOfInterest(
#     image.Image,
#     configuration_folder="features/images/vois/segmentation",
#     category="segmentation"
# ):
#     def __init__(self, **kwargs):
#         image.Image.__init__(self, **kwargs, modality="segmentation")
