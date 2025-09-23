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
Macaque receptor mapping and high-resolution MRI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


# %%
import siibra
from nilearn import plotting

# %%
space = siibra.spaces.get("mebrains")

# %%
fts = siibra.features.get(space, "mri")

# %%
tmpl_img = space.get_template().fetch(resolution_mm=-1)
plotting.view_img(
    fts[1].fetch(resolution_mm=-1),
    bg_img=tmpl_img,
    cmap="magma",
    symmetric_cmap=False,
    opacity=0.65,
)

# %%
voi = siibra.BoundingBox((-15.50, -21.50, -10.30), (1.70, 0.90, 0.10), space)
voi_img = fts[0].fetch(voi=voi, resolution_mm=-1)
plotting.view_img(
    voi_img,
    bg_img=None,
    cmap="gray",
    symmetric_cmap=False,
)

# %%
plotting.plot_img(voi_img, bg_img=None, cmap="gray", display_mode="y")
