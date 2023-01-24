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
from tqdm import tqdm

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
conn = features[0]
print(f"Connectivity features reflects {conn.modality} of {conn.cohort} cohort.")
print(conn.name)
print("\n" + conn.description)

# Subjects are encoded via anonymized ids:
print(conn.subjects)


# %%
# The connectivity matrices are provided as pandas DataFrames,
# with region objects as index.
subject = conn.subjects[0]
matrix = conn.get_matrix(subject)
matrix

# %%
# Alternatively, we can visualize the matrix using plot_matrix() method
conn.plot_matrix(subject=conn.subjects[0])
# %%
# The average matrix across all subjects can be displayed by leaving out subjects
# or setting it to "mean". This time let us display in logscale.
conn.plot_matrix(logscale=True)

# %%
# We can create a 3D visualization of the connectivity using
# the plotting module of `nilearn <https://nilearn.github.io>`_.
# To do so, we need to provide centroids in
# the anatomical space for each region (or "node") of the connectivity matrix.
# This requires is to 1) compute centroids, and 2) organize the centroids
# in the sequence of connectivity matrix rows.
#
# We start by computing the centroids for each region
# in the matrix as defined by its mask in MNI152 space.
# Note that we need to transform the centroids from voxels into
# physical space, using the affine matrix stored in the mask.
node_coords = [
    tuple(region.compute_centroids('mni152')[0])
    for region in tqdm(matrix.index)
]

# %%
# Now we can plot the structural connectome.
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
    node_size=3, colorbar=False,  edge_cmap="bwr"
    )

# %%
