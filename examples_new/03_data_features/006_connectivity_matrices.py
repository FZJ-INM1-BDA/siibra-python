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
Several types of connectivity are supported. As of now, these include
"StreamlineCounts", "StreamlineLengths", "FunctionalConnectivity", and
"AnatomoFunctionalConnectivity" (F-Tract).
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
# Since a single query may yield hundreds of connectivity matrices for different
# subjects of a cohort, siibra groups them as elements into
# :ref:`sphx_glr_examples_03_data_features_009_compound_features.py`.
# Let us select "HCP" cohort.
features = siibra.find_features(julich_brain, siibra.modality_types.STREAMLINECOUNTS)
for f in features:
    print(f)
    print(f.facets[:5])

# %%
# We can select a specific element by integer index
f = features[0]
filtered_f = f.filter_by_facets(subject="0114_1")
print(filtered_f.data)
print(filtered_f.aggregate_by)  # Subjects are encoded via anonymized ids

# %%
# The connectivity matrices are provided as pandas DataFrames, with region
# objects as indices and columns. We can access the matrix corresponding to
# the selected index by
matrix = filtered_f.data[0]
matrix.iloc[0:15, 0:15]  # let us see the first 15x15

# %%
# The matrix can be displayed using `plot` method. Here, we demonstrate showing
# a portion of the matrix with the first 30 rows and columns.
selected_regions = filtered_f.matrix_indices[0:15]
plotting.matrix_plotting(matrix.iloc[0:15, 0:15], labels=selected_regions, cmap="magma")

# %%
# We can create a 3D visualization of the connectivity using
# the plotting module of `nilearn <https://nilearn.github.io>`_.
# To do so, we need to provide centroids in
# the anatomical space for each region (or "node") of the connectivity matrix.
assert all(isinstance(idx, siibra.descriptions.RegionSpec) for idx in filtered_f.matrix_indices)
node_coords = [region.get_centroids().coordinates[0]
               for idx in filtered_f.matrix_indices
               if isinstance(idx, siibra.descriptions.RegionSpec)
               for region in idx.decode()]


# %%
# Now we can plot the structural connectome.
view = plotting.plot_connectome(
    adjacency_matrix=matrix,
    node_coords=node_coords,
    edge_threshold="80%",
    node_size=10,
)

aggregated_filter = "".join([f"{aggr.key}={aggr.value}" for aggr in filtered_f.aggregate_by])

view.title(
    f"{', '.join(filtered_f.modalities)} of"
    f"{aggregated_filter}"
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

# %%
# You can also access to F-Tract atlases using `siibra`. Unlike the most
# connectivity matrices in siibra, F-Tract matrices are subject-averaged but
# distinguished by their "feature".
ftract = siibra.find_features(
    siibra.parcellations.get("julich 3.0"),
    siibra.modality_types.ANATOMOFUNCTIONALCONNECTIVITY
)[0] 

print(ftract.name)
print("indexing attributes:", ftract.facets)

# %%
# We can similarly draw the data as above. This time let us get the element by
# its feature name
f = ftract.filter_by_facets(feature="P_11: speed__euclidian__peak_delay__median")
f.plot()
