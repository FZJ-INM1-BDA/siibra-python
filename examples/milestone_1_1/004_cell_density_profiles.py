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
Layer-specific cortical cell distributions and densities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
import siibra
from nilearn import plotting

# %%
# Cell density features for the human brain are collected for different
# cytoarchitectonic brain areas, each based on multiple cortical image patches
# extracted from 1 micron resolution BigBrain sections. The dataset consists of
# multiple derivatives for underlying image data, precise cortical layer
# annotations, cortical depth maps and the actual cell segmentations, expose as
# different feature modalities in siibra. One of these are layer-wise cell
# density statistics.
area_hoc6 = siibra.get_region("julich 3.1", "hoc6")
layerwise_cell_densities = siibra.features.get(area_hoc6, "layerwise cell density")
for lf in layerwise_cell_densities:
    print(lf.name)

# %%
# As tabular data, measurements are exposed by siibra as a DataFrame. Here,
# tables are indexed by cortical layer and provide statistics on cell counts,
# sizes, as well as layer thickness.
# TODO in a future version, expose all the infromation from the sidecars, which contain descriptionis for each column.
layerwise_cell_densities[0].data

# %%
# Columns of the table can be plotted in various ways, e.g. as box plot of
# density per layer.
layerwise_cell_densities[0].plot(
    y="cell_size_mean_um2", error_y="cell_size_std_um2", backend="plotly"
)

# %%
# Besides the layerwise summary statistics, individual cell density profiles can
# be accessed as well. The profiles are tables indexed by cortical depth and
# provide density and intensity statistics along the profile.
profiles = siibra.features.get(area_hoc6, "cell density profile")
profiles[0].data

# %%
# Profiles are best viewed as a plot,
profiles[0].plot(y="density_mean", error_y="density_std", backend="plotly", kind="line")

# %%
# The profiles are derived from detections of individual cells in each patch.
# These detailed measurements are similar to the cell distributions in rodent
# brains, and accessible through the "cell distibution" modality.
cell_dists = siibra.features.get(area_hoc6, siibra.features.cellular.CellDistribution)
cell_dists[0].data

# %%
# The data can be visualized as histograms across cell instances.
cell_dists[0].plot(
    column_name="area_um2",
    bins=100,
    xlabel=r"cell area $\mu m^2$",
    kind="hist",
)


# %%
# Since cell locations are points in atlas space, they can be loaded as
# `siibra.PointCloud` objects, and reused broadly by warping coordinates between
# Bigbrain,  MNI 152, and MNI Colin 27 reference spaces.
cell_locs = cell_dists[0].as_pointcloud()
bigbrain_tmpl_img_voi = siibra.get_template("bigbrain").fetch(
    voi=cell_locs.boundingbox, resolution_mm=-1
)
display = plotting.plot_img(
    img=bigbrain_tmpl_img_voi,
    cut_coords=cell_locs.centroid.coordinate,
    cmap="gray",
    display_mode="y",
)
display.add_markers(cell_locs.coordinates, marker_size=1)

# %%
# Warping points to the MNI 152 space is simple.
cell_locs_warped = cell_locs.warp("mni152")
display = plotting.plot_img(
    img=siibra.get_template("mni152").fetch(),
    cut_coords=cell_locs_warped.centroid.coordinate,
    cmap="gray",
    display_mode="y",
)
display.add_markers(cell_locs_warped.coordinates, marker_size=1)
