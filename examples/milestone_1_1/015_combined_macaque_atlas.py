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
Macaque Brain Composite Atlas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
# List all parcellations in siibra to see the new atlases
import siibra
from nilearn import plotting

for p in siibra.parcellations:
    print(p)

# %%
# A new Macaque atlas comprimised of receptor maps, retinotopy map, and CHARM
# atlas. The map is still under construction, hence, some metadata such as
# license and publication information are not present.
parcellation = siibra.parcellations["Macaque Brain Composite Atlas: MEBRAINS, fMRI, CHARM"]
print(parcellation.name)
print(parcellation.description)

# %%
# We can get the map as usual and also obtain the associated colormap and draw
# the map over the MEBRAINS template.
mp = parcellation.get_map("MEBRAINS")
cmap = mp.get_colormap()
img = mp.fetch()
template_img = siibra.get_template("MEBRAINS").fetch(resolution_mm=1)
plotting.plot_roi(
    img,
    bg_img=template_img,
    cmap=cmap,
    title=f"{mp.name}\nnumber of regions: {len(mp.regions)}",
    black_bg=False,
    cut_coords=(13, -5, 16),
    colorbar=False,
)

# %%
# The original colormap is designed to showcase the organisataion of the lower
# level structures. For a more detailed view, we can choose a discrete color map
# such as "paired":
plotting.plot_roi(
    img,
    bg_img=template_img,
    cmap="Paired",
    title=f"{mp.name}\nnumber of regions: {len(mp.regions)}",
    black_bg=False,
    cut_coords=(13, -5, 16),
    colorbar=False,
)
