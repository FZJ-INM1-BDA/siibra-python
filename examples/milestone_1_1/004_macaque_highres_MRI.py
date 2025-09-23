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
Macaque high-resolution MRI
~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


# %%
import siibra
from nilearn import plotting

# sphinx_gallery_thumbnail_path = '_static/example_thumbnails/macaque_high_res_MRI.png'

# %%
# Query for mri images registered on MEBRAINS population-based monkey template
space = siibra.spaces.get("mebrains")
mri_images = siibra.features.get(space, "mri")
for f in mri_images:
    print(f.name)
    print("Publication:", f.urls)

# %%
# Fetch the template image and overlay 180um resoltion image.
tmpl_img = space.get_template().fetch(resolution_mm=-1)
mri_180um = [f for f in mri_images if "180micron" in f.name][0]
plotting.view_img(
    mri_180um.fetch(resolution_mm=-1),
    bg_img=tmpl_img,
    cmap="magma",
    symmetric_cmap=False,
    opacity=0.65,
)

# %%
# Or plot without background in full resolution since Mebrains macaque template
# has 40um resolution
plotting.view_img(
    mri_180um.fetch(resolution_mm=-1),
    bg_img=None,
    cmap="gray",
    symmetric_cmap=False,
    black_bg=True,
)

# %%
# Now, select a volume of interest and plot 100um resoltion MRI
voi = siibra.BoundingBox((-15.50, -21.50, -10.30), (1.70, 0.90, 0.10), space)
mri_100um = [f for f in mri_images if "100micron" in f.name][0]
voi_img = mri_100um.fetch(voi=voi, resolution_mm=-1)
plotting.view_img(
    voi_img,
    bg_img=None,
    cmap="gray",
    symmetric_cmap=False,
)
