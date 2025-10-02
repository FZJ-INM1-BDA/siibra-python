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
dMRI streamline matrices - 1000 Brains cohort
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
import siibra
import matplotlib.pyplot as plt
from nilearn import plotting

# %%
# Connectivity matrices are averaged over regions from a reference parcellation.
# Here, Julich Brain is used to query for streamline counts. The query returns
# datasets based on two different cohorts.
julich_brain = siibra.parcellations["julich 3.1"]
streamline_count_matrices = siibra.features.get(julich_brain, "streamlinecounts")
for sc in streamline_count_matrices:
    print(sc.cohort)
# %%
# Now, filter out the 1000BRAINS cohort and check the subject fields. This
# dataset is constructed for showcasing group averages and we will rely on this
sc = list(filter(lambda f: f.cohort == "1000BRAINS", streamline_count_matrices))[0]
print(sc.name)
for s in sc:
    print("matrix:", s.subject)

# %%
hoc1left = julich_brain.get_region("hoc1 left")
fig, axs = plt.subplots(1, 7, sharey=True)
fig.set_size_inches(15, 6)
for i, grp in enumerate(sc):
    if "age" not in grp.subject:
        continue
    grp.plot(
        regions=hoc1left,
        backend="matplotlib",
        min_connectivity=500,
        max_rows=10,
        ax=axs[i],
        title=grp.subject,
    )

# %%
hoc1right = julich_brain.get_region("hoc1 right")
fig, axs = plt.subplots(1, 7, sharey=True)
fig.set_size_inches(15, 6)
for i, grp in enumerate(sc):
    if "age" not in grp.subject:
        continue
    grp.plot(
        regions=hoc1right,
        backend="matplotlib",
        min_connectivity=500,
        max_rows=10,
        logscale=True,
        ax=axs[i],
        title=grp.subject,
    )


# %%
youngest_group = sc[0]
node_coords = youngest_group.compute_centroids("mni152")
plotting.view_connectome(
    adjacency_matrix=youngest_group.data,
    node_coords=node_coords,
    edge_threshold="99%",
    node_size=3,
    colorbar=False,
    edge_cmap="bwr",
)
