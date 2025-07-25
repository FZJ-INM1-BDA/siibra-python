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
New Maps and Atlases
~~~~~~~~~~~~~~~~~~~~
"""

# %%
import siibra
from nilearn import plotting
import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_path = '_static/example_thumbnails/milestone_1_1_macaque_combined_atlas.png'

# %%
combined_macaque = siibra.parcellations.get("combined macaque atlas")
combined_macaque.render_tree()

# %%
combined_macaque_map = combined_macaque.get_map("mebrains")
plotting.view_img(
    combined_macaque_map.fetch(),
    bg_img=combined_macaque_map.space.get_template().fetch(),
    cmap=combined_macaque_map.get_colormap(),
    symmetric_cmap=False,
)


# %%
sulci_atlas = siibra.parcellations.get("sulci atlas")
sulci_atlas.render_tree()

# %%
cut_coords = siibra.Point((50, 9, 34), "colin27")
fig, axs = plt.subplots(3, 1, figsize=(15, 21))
for i, space in enumerate(
    siibra.maps.dataframe.query(f'parcellation == "{sulci_atlas.name}"')["space"]
):
    mp = sulci_atlas.get_map(space)
    cmap = mp.get_colormap()
    sulci_img = mp.fetch(max_bytes=0.4 * 1024**3)
    template_img = mp.space.get_template().fetch(
        resolution_mm=sulci_img.header.get_zooms()[0], max_bytes=0.4 * 1024**3
    )
    plotting.plot_img(
        sulci_img,
        bg_img=template_img,
        cmap=cmap,
        title=space,
        cut_coords=cut_coords.warp(space).coordinate,
        resampling_interpolation="nearest",
        axes=axs[i],
    )


# %%
marmoset = siibra.parcellations.get("marmoset")
print(marmoset.publications)
marmoset.render_tree()

# %%
marmoset_map = marmoset.get_map("marmoset")
plotting.view_img(
    marmoset_map.fetch(),
    bg_img=marmoset_map.space.get_template().fetch(),
    cmap=marmoset_map.get_colormap(),
    symmetric_cmap=False,
    resampling_interpolation="nearest",
)
