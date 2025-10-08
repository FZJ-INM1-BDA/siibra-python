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
Macaque neurotransmitter receptor density
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


# %%
import siibra
from nilearn import plotting

# sphinx_gallery_thumbnail_number = -1

# %%
# Query for tabular features anchored in MEBRAINS atlas
MEBRAINS = siibra.parcellations["MEBRAINS monkey"]
for f in siibra.features.get(MEBRAINS, siibra.features.generic.Tabular):
    print("Modality:", f.modality)

# %%
# Specify receptor density modality
receptor_density = siibra.features.get(
    MEBRAINS,
    siibra.features.generic.Tabular,
    modality="Neurotransmitter receptor density",
)[0]
print(receptor_density.description)
print("".join(receptor_density.urls))

# %%
# Fetch the data as a DataFrame with areas on the rows and receptors on the
# columns
receptor_density.data

# %%
# For a closer look, select a receptor
receptor = "kainate"

# %%
# Plot the data for receptor "kainate". (Similarly, one can plot it for other
# receptors).
receptor_density.plot(
    kind="barh",
    y=receptor,
    ylabel="area",
    xlabel=f"{receptor} - {receptor_density.unit}",
    figsize=(10, 20),
)

# %%
# Now, decipher the published masks of MEBRAINS regions by iterating over
# MEBRAINS map:
MEBRAINS_map = MEBRAINS.get_map("MEBRAINS")
mapped_regions = {}
for r in receptor_density.data.index:
    try:
        mapped_regions[r] = MEBRAINS.get_region(r.replace("area_", "")).name
    except ValueError:
        print(f"Mask of {r} is not yet published.")

# %%
# Using `colorize()` method, we can color the regional masks with values in the
# table for selected receptor. `view_img` allows us to view it dynamically.
receptor_values = {}
for r, v in receptor_density.data[receptor].items():
    if r in mapped_regions:
        receptor_values[mapped_regions[r] + " left"] = v
        receptor_values[mapped_regions[r] + " right"] = v
receptor_density_map = MEBRAINS_map.colorize(receptor_values, resolution_mm=-1)
plotting.view_img(
    receptor_density_map.fetch(),
    bg_img=MEBRAINS_map.space.get_template().fetch(),
    cmap="magma",
    symmetric_cmap=False,
    resampling_interpolation="nearest",
)

# %%
# For demonstration, plot the all recoptor densities as static plots at the
# centroid of area 8Bs left.

# compute the centroid of 8Bs left
centroid = (
    MEBRAINS_map.parcellation.get_region("8Bs left")
    .compute_centroids("MEBRAINS")[0]
    .coordinate
)
# compute the max and min values of the intensities for a standard colarbar across the plots
vmin = receptor_density.data.min().min()
vmax = receptor_density.data.max().max()

# color the map and plot the image per receptor
for i, receptor in enumerate(receptor_density.data.columns):
    receptor_values = {}
    for r, v in receptor_density.data[receptor].items():
        if r in mapped_regions:
            receptor_values[mapped_regions[r] + " left"] = v
            receptor_values[mapped_regions[r] + " right"] = v
    receptor_density_map = MEBRAINS_map.colorize(receptor_values, resolution_mm=-1)
    plotting.plot_stat_map(
        receptor_density_map.fetch(),
        bg_img=MEBRAINS_map.space.get_template().fetch(),
        cmap="magma",
        resampling_interpolation="nearest",
        cut_coords=centroid,
        title=receptor,
        vmin=vmin,
        vmax=vmax,
    )
