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
# connectivity matrices for different files from an imaging cohort. In most
# cases, these correspond to subjects, like in this example.
conn = features[0]
print(f"Connectivity features reflects {conn.modality} of {conn.cohort} cohort.")
print(conn.name)
print("\n" + conn.description)

# Subjects are encoded via anonymized ids:
print([f.matrix_keys for f in conn])
subject = '188'  # let's select subject 188

# %%
# The connectivity matrices are provided as pandas DataFrames,
# with region objects as index.
matrix = conn[subject].data
matrix

# You can also obtain the average by `average_matrix = conn.data`.

# %%
# Alternatively, we can visualize the matrix using plot() method
# for each subject
conn.plot(subject)

# %%
# If interested in the profile of a region we can simply plot by
conn.plot(subject, regions="hoc1 left", backend='plotly')

# %%
# The average matrix across all subjects can be displayed by leaving out matrix
# key or setting it to `None`. Also, the matrix can be displayed by specifiying
# a list of regions.
selected_regions = conn[subject].regions[0:30]
conn.plot(regions=selected_regions, reorder=True, cmap="magma") 

# %%
# We can create a 3D visualization of the connectivity using
# the plotting module of `nilearn <https://nilearn.github.io>`_.
# To do so, we need to provide centroids in
# the anatomical space for each region (or "node") of the connectivity matrix.
node_coords = conn[subject].compute_centroids('mni152')

# %%
# Now, using nilearn, we can also plot the structural connectome.
view = plotting.plot_connectome(
    adjacency_matrix=matrix,
    node_coords=node_coords,
    edge_threshold="80%",
    node_size=10,
)
view.title(
    f"{conn.modality} of subject {subject} in {conn.cohort} cohort "
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
