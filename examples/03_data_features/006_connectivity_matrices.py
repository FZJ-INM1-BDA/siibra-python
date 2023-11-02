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
julich_brain = siibra.parcellations.get("julich 2.9")

# %%
# The matrices are queried as expected, using `siibra.features.get`, passing
# the parcellation as a concept. Here, we query for structural connectivity matrices.
# Connectivity features are compounded into a big set which contain the connectivity
# matrices as elements. Typically, connectivity features provide a range
# of region-to-region connectivity matrices for different subjects from an
# imaging cohort. Let us select "HCP" cohort.
features = siibra.features.get(julich_brain, siibra.features.connectivity.StreamlineCounts)
for cf in features:
    print(cf.name)
    if cf.compounding_attributes["cohort"] == "HCP":
        conn_cf = cf
print(f"Selected: {conn_cf.name}'\n'" + conn_cf.description)

# %%
# A specific `StreamlineCounts` object expresses structural connectivity in
# the form of numbers of streamlines connecting pairs of brain regions as
# estimated from tractography on diffusion imaging. We can access to these
# elements by integers or the element indices. In this case indicies represent
# subjects.
for i, cf in enumerate(conn_cf[:10]):  # we can iterate over elements of a CompoundFeature
    print(f"Element index: {conn_cf.indices[i]}, Subject: {cf.subject}")

# %%
# We can select a specific element by integer index or element index via
# `get_element` method
print(conn_cf[5].name)
print(conn_cf.get_element('005').name)

# %%
# The connectivity matrices are provided as pandas DataFrames, with region
# objects as indices and columns. We can access the matrix corresponding to
# the selected index by
matrix = conn_cf[0].data
matrix.iloc[0:15, 0:15]  # let us see the first 15x15

# %%
# The matrix can be displayed using `plot` method. In addition, it can be
# displayed only for a specific list of regions.
selected_regions = conn_cf[0].regions[0:30]
conn_cf[0].plot(regions=selected_regions, reorder=True, cmap="magma")

# %%
# We can create a 3D visualization of the connectivity using
# the plotting module of `nilearn <https://nilearn.github.io>`_.
# To do so, we need to provide centroids in
# the anatomical space for each region (or "node") of the connectivity matrix.
node_coords = conn_cf[0].compute_centroids('mni152')


# %%
# Now we can plot the structural connectome.
view = plotting.plot_connectome(
    adjacency_matrix=matrix,
    node_coords=node_coords,
    edge_threshold="80%",
    node_size=10,
)
view.title(
    f"{conn_cf.modality} of subject {conn_cf.indices[0]} in {conn_cf[0].cohort} cohort "
    f"averaged on {julich_brain.name}",
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
