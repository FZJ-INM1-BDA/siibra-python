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

from .image.nifti import fetch_nifti
from .image.neuroglancer import fetch_neuroglancer
from .mesh.gifti import fetch_gii_mesh, fetch_gii_label
from .mesh.neuroglancer import fetch_neuroglancer_mesh
from .mesh.freesurfer import fetch_freesurfer_annot
from .volume_fetcher import (
    FetchKwargs,
    Mapping,
    MESH_FORMATS,
    IMAGE_FORMATS,
    SIIBRA_MAX_FETCH_SIZE_GIB,
)
