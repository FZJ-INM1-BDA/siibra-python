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
import siibra
from nilearn import plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# %%
# List all parcellations in siibra to see the new atlases
for p in siibra.parcellations:
    print(p)

# %%
# Sulci atlas is another new addition to siibra. We can display associated
# description, license, and urls/dois
sulci_parcellation = siibra.parcellations["Sulci atlas"]
print(sulci_parcellation.name)
print(sulci_parcellation.description)
print(sulci_parcellation.urls)
print(sulci_parcellation.LICENSE)

# %%
# The human sulci atlas was mapped in several spaces. Loop through the map
# registry while filtering for this parcellation and fetch the images
# to plot them on their respective reference templates.
cut_coords = siibra.Point((11, 52, 30), "mni 152")  # define the coordinate to view the maps

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

# %%
# Additionally, majority of the sulci can be fetched as surfaces. To illustrate,
# draw the surfaces along with the colin 27 template
mp = sulci_parcellation.get_map("colin 27")
meshes = {
    r: mp.fetch(r, format="mesh")
    for r in mp.regions
    if mp.fetch(r, format="mesh") is not None
}
labels = [mp.get_index(r).label for r in meshes.keys()]
template_mesh = mp.space.get_template().fetch(format="mesh")

# combine meshes for display engine
mesh = siibra.commons.merge_meshes(
    meshes=[template_mesh] + list(meshes.values()),
    labels=[0] + labels
)

# create custom color map
base_cmap = plt.get_cmap("Paired")
custom_colors = [(1, 1, 1, 0.9)] + [
    base_cmap(lb / max(labels)) for lb in labels
]
custom_cmap = mcolors.ListedColormap(custom_colors)

# plot using nilearn
plotting.plot_surf(
    [mesh["verts"], mesh["faces"]],
    mesh["labels"],
    engine="plotly",
    cmap=custom_cmap,
    colorbar=False,
)
