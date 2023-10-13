# Copyright 2018-2021
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
Connectivity matrices
~~~~~~~~~~~~~~~~~~~~~

`siibra` provides access to parcellation-averaged connectivity matrices.
Several types of connectivity are supported.
As of now, these include "StreamlineCounts", "StreamlineLengths", and "FunctionalConnectivity".
"""

# %%
from nilearn import plotting
import siibra
# sphinx_gallery_thumbnail_number = 2

# %%
# We start by selecting an atlas parcellation.
jubrain = siibra.parcellations.get("julich 2.9")

# %%
# The matrices are queried as expected, using `siibra.features.get`,
# passing the parcellation as a concept.
# Here, we query for structural connectivity matrices.
features = siibra.features.get(jubrain, siibra.features.connectivity.StreamlineCounts)
print(f"Found {len(features)} streamline count matrices.")

# %%
# We fetch the first result, which is a specific `StreamlineCounts` object
# expressing structural connectivity in the form of numbers of streamlines
# connecting pairs of brain regions as estimated from tractography on diffusion imaging.
# Typically, connectivity features provide a range of region-to-region
# connectivity matrices for different subjects from an imaging cohort.
print("Connectivity features are compounded by the modality and cohort.")
for cf in features:
    print(cf.name)
    # let us select the HCP cohort
    if "HCP" in cf.compounding_attritbutes:
        conn = cf

print(f"Selected: {conn.name}'\n'" + conn.description)

# %%
# The connectivity matrices are provided as pandas DataFrames,
# with region objects as index. We can access to the average by calling `data`
# property from the CompoundFeature.
conn.data.iloc[0:15, 0:15]  # see the first 15x15 for simplicity

# %%
# Subjects are encoded via anonymized ids
print(conn.indices)
index = conn.indices[0]  # let us select the first subject

# %%
# we can access to corresponding matrix via
matrix = conn[index].data
matrix.iloc[0:15, 0:15]  # let us see the first 15x15

# %%
# The matrix can be displayed using `plot` method. Also, it can be
# displayed only for a specific list of regions.
selected_regions = conn[index].regions[0:30]
conn[index].plot(regions=selected_regions, reorder=True, cmap="magma")

# %%
# We can create a 3D visualization of the connectivity using
# the plotting module of `nilearn <https://nilearn.github.io>`_.
# To do so, we need to provide centroids in
# the anatomical space for each region (or "node") of the connectivity matrix.
node_coords = conn[index].compute_centroids('mni152')


# %%
# Now we can plot the structural connectome.
view = plotting.plot_connectome(
    adjacency_matrix=matrix,
    node_coords=node_coords,
    edge_threshold="80%",
    node_size=10,
)
view.title(
    f"{conn.modality} of subject {index} in {conn[index].cohort} cohort "
    f"averaged on {jubrain.name}",
    size=10,
)

# %%
# or in 3D:
plotting.view_connectome(
    adjacency_matrix=matrix,
    node_coords=node_coords,
    edge_threshold="99%",
    node_size=3, colorbar=False,
    edge_cmap="bwr"
)
