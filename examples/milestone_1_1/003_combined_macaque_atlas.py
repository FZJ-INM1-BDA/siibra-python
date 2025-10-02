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
Combined Macaque Brain Atlas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
# List all parcellations in siibra to see the new atlases
import siibra
from nilearn import plotting

for p in siibra.parcellations:
    print(p)

# %%
# There are 3 new parcellations. We make a list from the parcellations
# and display associated description, license, and urls/dois
parcellation = siibra.parcellations["Combined Macaque Brain Atlas: MEBRAINS, fMRI, CHARM"]
print(parcellation.name)
print(parcellation.description)
print(parcellation.urls)
print(parcellation.LICENSE)

# %%
# Some parcellations are mapped in several spaces. Therefore, loop through the
# map registry while filtering for these parcellations. Then, fetch the images
# for each and plot them on their respective space templates.
mp = parcellation.get_map("mebrains")
cmap = mp.get_colormap()
fetch_kwargs = {"max_bytes": 0.4 * 1024 ** 3} if "neuroglancer/precomputed" in mp.formats else {}
img = mp.fetch(**fetch_kwargs)
template_img = siibra.get_template("mebrains").fetch(resolution_mm=1)
plotting.plot_roi(
    img,
    bg_img=template_img,
    cmap=cmap,
    title=f"{mp.name}\nnumber of regions: {len(mp.regions)}",
    black_bg=False,
    cut_coords=(13, -5, 16),
    colorbar=False,
)
