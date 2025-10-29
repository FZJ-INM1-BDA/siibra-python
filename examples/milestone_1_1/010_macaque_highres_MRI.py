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

"""
High-resolution structural MRI of themacaque brain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


# %%
import siibra
from nilearn import plotting

# sphinx_gallery_thumbnail_path = '_static/example_thumbnails/macaque_high_res_MRI.png'

# %%
# We search for MRI's anchored to the MEBRAINS reference space,
# and generate a tabular overview.
space = siibra.spaces.get("MEBRAINS")
features_mri = siibra.features.get(space, "mri")
siibra.features.tabulate(
    features_mri, ['name', 'urls'],
    converters={'urls': lambda u: ','.join(u)}
)

# %%
# We fetch thhe 180um resolution image, resulting in a Nifti1Image object
# that can be plotted with the template using common tools such as nilearn's 
# plotting module.
# TODO this could use a better colormap or vmax setting
tmpl_img = space.get_template().fetch(resolution_mm=-1)
mri_180um = features_mri[1]
plotting.view_img(
    mri_180um.fetch(resolution_mm=-1),
    bg_img=tmpl_img,
    cmap="magma",
    symmetric_cmap=False,
    opacity=0.65,
)


# %%
# To take advantage of the high resolution while 
# avoiding to download the full high-resolution volume,
# a region of interest can  be used for fetching.
voi = siibra.BoundingBox((-15.50, -21.50, -10.30), (1.70, 0.90, 0.10), space)
mri_100um = features_mri[0]
voi_img = mri_100um.fetch(voi=voi, resolution_mm=-1)
plotting.view_img(
    voi_img,
    bg_img=None,
    cmap="gray",
    symmetric_cmap=False,
)
