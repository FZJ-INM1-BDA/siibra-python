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
dMRI streamline counts - 1000 Brains cohort
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
streamline_counts = siibra.features.get(julich_brain, "streamlinecounts")
for sc in streamline_counts:
    print(sc.cohort)
# %%
# After filtering out the 1000BRAINS cohort and we check the subject fields to
# observe that the individual matrices corresponds to group of subjects.
sc = [f for f in streamline_counts if f.cohort == "1000BRAINS"][0]
print(sc.name)
for s in sc:
    print("matrix:", s.subject)

# %%
# We select a region to compare connectivity profiles among different age groups.
area_hoc1_l = julich_brain.get_region("hoc1 left")
fig, axs = plt.subplots(1, 7, sharey=True)
fig.set_size_inches(15, 6)
for i, grp in enumerate(sc):
    if "age" not in grp.subject:
        continue
    grp.plot(
        regions=area_hoc1_l,
        min_connectivity=500,
        max_rows=10,
        ax=axs[i],
        title=grp.subject,
    )

# %%
# We then do the same comparison for area 44 right hemisphere in order to
# observe the trends over age groups.
area_44_r = julich_brain.get_region("44 right")
fig, axs = plt.subplots(1, 7, sharey=True)
fig.set_size_inches(15, 6)
for i, grp in enumerate(sc):
    if "age" not in grp.subject:
        continue
    grp.plot(
        regions=area_44_r,
        min_connectivity=500,
        max_rows=10,
        ax=axs[i],
        title=grp.subject,
    )


# %%
# Alternatively, using nilearn's connectome viewer, the connectivity matrices
# can be explored in MNI 152 surface where the node coordinates refer to
# centroids of the regions
female_group = [f for f in sc if f.subject == "sex-group-mean-female"][0]
node_coords = female_group.compute_centroids("mni152")
plotting.view_connectome(
    adjacency_matrix=female_group.data,
    node_coords=node_coords,
    edge_threshold="99%",
    node_size=2,
)
