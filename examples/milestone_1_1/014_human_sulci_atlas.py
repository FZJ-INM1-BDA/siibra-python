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
Human sulci atlas
~~~~~~~~~~~~~~~~~
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
sulci_parcellation = siibra.parcellations["Sulci atlas"]
print(sulci_parcellation.name)
print(sulci_parcellation.description)
print(sulci_parcellation.urls)
print(sulci_parcellation.LICENSE)

# %%
# Some parcellations are mapped in several spaces. Therefore, loop through the
# map registry while filtering for these parcellations. Then, fetch the images
# for each and plot them on their respective space templates.

# define coordinates to view the maps
cut_coords = siibra.Point((11, 52, 30), "mni 152")
# plot each in a subplot
mapped_spaces = siibra.maps.dataframe.query(
    f'parcellation == "{sulci_parcellation.name}"'
)["space"]
for mp in siibra.maps:
    if mp.parcellation != sulci_parcellation:
        continue
    cmap = mp.get_colormap()
    fetch_kwargs = (
        {"max_bytes": 0.4 * 1024**3} if "neuroglancer/precomputed" in mp.formats else {}
    )
    img = mp.fetch(**fetch_kwargs)
    template_img = siibra.get_template(mp.space).fetch(resolution_mm=1)
    plotting.plot_roi(
        img,
        bg_img=template_img,
        cmap=cmap,
        title=mp.space.name,
        black_bg=False,
        cut_coords=cut_coords.warp(mp.space).coordinate,
        colorbar=False,
    )

