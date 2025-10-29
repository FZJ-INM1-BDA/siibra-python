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
Neurotransmitter receptor densities in macaque brains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


# %%
import siibra
from nilearn import plotting
from collections import defaultdict

# sphinx_gallery_thumbnail_number = -1

# %%
# The Macaque atlas is a rather new development in siibra. 
# We compile an overview of  tabular features linked with the
# MEBRAINS parcellation by modality, and find that the tabular
# features include neurotransmitter receptor density data.
mebrains = siibra.parcellations.get("MEBRAINS monkey")
features_tabular = siibra.features.get(mebrains, siibra.features.generic.Tabular)
siibra.features.tabulate(features_tabular, ['modality'])

# %%
# If known, the modality name could be used to refine the tabular feature query.
features_rd = siibra.features.get(
    mebrains,
    siibra.features.generic.Tabular,
    modality="Neurotransmitter receptor density",
)

# %%
# The receptor density feature comes with the usual metadata and DOI links.
receptor_density = features_rd[0]
print(receptor_density.description)
print("".join(receptor_density.urls))

# %%
# Densities are exposed in the usual pandas DataFrame format,
# with rows representing brain areas and columns representing different receptors.
receptor_density.data

# %%
# Plot the data for the specific receptor. 
receptor = "kainate"
receptor_density.plot(
    kind="barh",
    y=receptor,
    ylabel="area",
    xlabel=f"{receptor} - {receptor_density.unit}",
    figsize=(10, 20),
)


# %%
# The data can be used to colorize the parcels of the MEBRAINS parcellation.
# We have to consider that not every region with available densities is yet 
# mapped in the parcellation.
# The list of mapped regions is stored for re-use below.
kainate_densities = {}
MEBRAINS_map = mebrains.get_map("MEBRAINS")

# collect densities for mapped brain regions
mapped_regions = defaultdict(list)
for regionspec, density in receptor_density.data[receptor].items():
    try:
        region = mebrains.get_region(regionspec.split('_')[-1])
        for child in region.leaves:  # colorize the child nodes
            kainate_densities[child.name] = density
            mapped_regions[regionspec].append(child)
    except ValueError:
        continue  # region not yet mapped

# plot the colorized map
kainate_map = MEBRAINS_map.colorize(kainate_densities, resolution_mm=-1)
plotting.view_img(
    kainate_map.fetch(),
    bg_img=MEBRAINS_map.space.get_template().fetch(),
    cmap="magma",
    symmetric_cmap=False,
    resampling_interpolation="nearest",
)

# %%
# Plot the all receptor densities as static plots.

# Use the centroid of 8Bs left to position the crosshair.
cut_coord = (
    mebrains.get_region("8Bs left")
    .compute_centroids("MEBRAINS")[0]
    .coordinate
)

# color the map and plot the image per receptor
vmax = receptor_density.data.max().max()
for i, receptor in enumerate(receptor_density.data.columns):

    # collect density of this receptor per mapped region
    receptor_densities = {}
    for regionspec, density in receptor_density.data[receptor].items():
        for region in mapped_regions[regionspec]:
            receptor_densities[region.name] = density

    # plot the map
    receptor_map = MEBRAINS_map.colorize(receptor_densities, resolution_mm=-1)
    plotting.plot_stat_map(
        receptor_map.fetch(),
        bg_img=MEBRAINS_map.space.get_template().fetch(),
        cmap="magma",
        resampling_interpolation="nearest",
        cut_coords=cut_coord,
        title=receptor,
        vmax=vmax,
    )

# %%
