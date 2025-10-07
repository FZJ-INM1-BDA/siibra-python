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
Cell distributions in human brain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
import siibra
from nilearn import plotting

# %%
# The cell density statistics are collected for each region in Julich Brain
# from 1 micron bigbrain sections across several different chunks. The
# accompanying information is pulled directly from BIDS-like dataset
# structure and file names. The most concise statistics concerning this dataset
# is the statistics per layer which can be queried with a region and the
# "layerwise cell density" modality.
area_hoc6 = siibra.get_region("julich 3.1", "hoc6")
layerwise_cell_densities = siibra.features.get(area_hoc6, "layerwise cell density")
for lf in layerwise_cell_densities:
    print(lf.name)

# %%
# The data is represented as pandas DataFrame where layers are indicies and
# cell statistics in columns.
layerwise_cell_densities[0].data
# %%
# The desired statistics, in this case cell size, can be plotted with ease
# from the feature by means of the integrated pandas plotting backend in siibra
# by passing the usual keywords
layerwise_cell_densities[0].plot(
    y="cell_size_mean_um2", error_y="cell_size_std_um2", backend="plotly"
)

# %%
# More detailed data is present in the cell density profiles, whice are
# integrated as pandas DataFrames indexed with cortical depth and contain
# density and intensity statistics as well as the cortical layer information.
profiles = siibra.features.get(area_hoc6, "cell density profile")
profiles[0].data

# %%
# We use the same mechanism above to plot the density profile
profiles[0].plot(y="density_mean", error_y="density_std", backend="plotly", kind="line")

# %%
# Even more detailed data, namely, the cell locations in the patches can be
# obtained by cell distribution modality features. The data is primarly
# represented as a DataFrame and contains the locations in the patch and size
# along with the coordinates in the BigBrain.
cell_dists = siibra.features.get(area_hoc6, siibra.features.cellular.CellDistribution)
cell_dists[0].data

# %%
# We can use this data to draw the distribution of cell areas easily similar
# to above
cell_dists[0].plot(
    column_name="area_um2",
    bins=100,
    xlabel=r"cell area $\mu m^2$",
    kind="hist",
)


# %%
# Moreover, the cell locations can be converted into `siibra.PointCloud` in
# order to make use of other functionalities such as warping coordinates between
# Bigbrain,  MNI 152, and MNI Colin 27 reference spaces. We simply represent the
# cells as a PointCloud and mark them on the BigBrain template
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
# Warp the PointCloud to MNI 152 space and display the local coordinates on
# the template with nilearn
cell_locs_warped = cell_locs.warp("mni152")
display = plotting.plot_img(
    img=siibra.get_template("mni152").fetch(),
    cut_coords=cell_locs_warped.centroid.coordinate,
    cmap="gray",
    display_mode="y",
)
display.add_markers(cell_locs_warped.coordinates, marker_size=1)
