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
Parcellation-averaged sructural connectivity from diffusion MRI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
import siibra
import matplotlib.pyplot as plt
from nilearn import plotting

# %%
# Querying with a reference parcellation yields connectivity matrices
# with information aggregated over brain areas.
# For streamline counts, matrix elements then encode numbers of 
# streamlines observed between two brain areas.
# The query for Julich-Brain results in multiple matrices, 
# referring to subjects or subject groups from different cohorts.
# Matrices of the same cohort and dataset are collected into compound features.
# Here, the compound for the "1000BRAINS" cohort is selected.
julich_brain = siibra.parcellations["julich 3.1"]
features = siibra.features.get(julich_brain, "streamlinecounts")
for f in features:
    print(f.cohort)
    if f.cohort == "1000BRAINS":
        sc = f
print(sc.name)

# %%
# The matrices in this compound refer to age and sex groups: 
for s in sc:
    print(s.subject)

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
