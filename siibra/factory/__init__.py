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

"""
This subpackage (siibra.factory) is where siibra fetches and builds siibra objects from various sources.

- .configuration.Configuration reads preconfigured/foundational siibra specificiations from an archive (e.g. git repository).
It uses factory to construct siibra-instances.

- livequery.* connects to external repositories to generate instances of siibra

- userfunctions (NYI) should be the place where user inputs
"""

# needed to boostrap preconfigured objects

from .configuration import iter_preconfigured
from .livequery import iter_livequery_clss

# import convenient image builders
from ..attributes.datarecipes.volume.image import (
    from_nifti as imageprovider_from_nifti,
    from_pointcloud as imageprovider_from_pointcloud,
    from_array as imageprovider_from_array,
)

# import convenient location builders

from ..attributes.locations.pointcloud import (
    sample_from_image as pointcloud_sampled_from_image,
    peaks_from_image as pointcloud_from_image_peaks,
)
