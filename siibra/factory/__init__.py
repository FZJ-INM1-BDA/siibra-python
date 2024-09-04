# Copyright 2018-2024
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

# needed to boostrap preconfigured objects
from . import configuration

from .factory import build_feature, build_object
from .iterator import iter_preconfigured_ac
from .livequery import iter_livequery_clss

from ..attributes.dataproviders.volume.image import from_nifti as imageprovider_from_nifti
from ..attributes.locations.pointcloud import (
    sample_from_image as pointcloud_sampled_from_image,
    peaks_from_image as pointcloud_from_image_peaks
)