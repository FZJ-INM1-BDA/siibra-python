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

`siibra` provides access to connectivity matrices from different sources containing averaged connectivity information for brain parcellations.
These matrices are modelled as ParcellationFeature types, so they match against a complete parcellation.
Several types of connectivity are supported.
As of now, these include the feature modalities "StreamlineCounts", "StreamlineLengths", and "FunctionalConnectivity".
"""

from nilearn import plotting
import siibra


# %%
# We start by selecting an atlas parcellation.
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
jubrain = atlas.get_parcellation("julich 2.9")

# %%
# The matrices are queried as expected, using `siibra.get_features`, and passing the parcellation as a concept.
# Here, we query for structural connectivity matrices.
features = siibra.get_features(jubrain, siibra.modalities.StreamlineCounts)
print(f"Found {len(features)} streamline count matrices.")

# %%
# We fetch the first result and have a look at it.
# It is a specific `StreamlineCounts` object, but it is derived from the more general `ConnectivityMatrix` class.
# The `src_info` attribute contains more detailed metadata information about each matrix.
conn = features[0]
print(f"Matrix reflects {conn.modality()} for subject {conn.subject} of {conn.cohort}.")
print("\n" + "; ".join(conn.authors))
print(conn.name)
print("Dataset id: " + conn.dataset_id)
print("\n" + conn.description)


# %%
# Connectivity matrix objects provide a pandas DataFrame for the connectivity measures,
# with full region objects as index.
conn.matrix

# %%
# We can create a 3D visualization of the connectivity using
# the plotting module of `nilearn <https://nilearn.github.io>`_.
# To do so, we need to provide centroids in
# the anatomical space for each region (or "node") of the connectivity matrix.
# This requires is to 1) compute centroids, and 2) organize the centroids
# in the sequence of connectivity matrix rows.
#
# We start by computing the centroids for each region in the parcellation map in MNI152 space.
parcmap = jubrain.get_map(space="mni152")
centroids = parcmap.compute_centroids()

# %%
# The centroids are a dictionary by region object, and each centroid is a proper siibra.Point object.
# We look at the first pair.
region, centroid = next(iter(centroids.items()))
print(region)
print(centroid)

# %%
# We extract a list of coordinate tuples by
# decoding each connectivity matrix region into the corresponding region of the parcellation map,
# and then fetching the coordinate tuple of its corresponding centroid.
node_coords = [tuple(centroids[parcmap.get_region(r)]) for r in conn.matrix.index]

# %%
# Now we can plot the structural connectome.
view = plotting.plot_connectome(
    adjacency_matrix=conn.matrix,
    node_coords=node_coords,
    edge_threshold="80%",
    node_size=10,
)
view.title(
    f"{conn.modality()} {conn.src_info['cohort']}/{conn.src_info['subject']} on {jubrain.name}",
    size=10,
)
