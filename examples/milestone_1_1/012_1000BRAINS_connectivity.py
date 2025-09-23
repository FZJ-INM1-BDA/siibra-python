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
# We start by loading the library
import siibra

# %%
# We choose a cortical region from Julich Brain and find fiber bundles
# overlapping with this region
julich_brain = siibra.parcellations["julich 3.1"]
area3b_left = julich_brain.get_region("Area 3b (PostCG) left")
# %%
# Next we will utilize streamline count matrices for further analysis
streamline_count_matrices = siibra.features.get(julich_brain, "streamlinecounts")
for sc in streamline_count_matrices:
    print(sc.cohort)
# %%
# Now, filter out the 1000BRAINS cohort and check the subject fields. This
# dataset is constructed for showcasing group averages and we will rely on this
sc = list(filter(lambda f: f.cohort == "1000BRAINS", streamline_count_matrices))[0]
print(sc.name)
for s in sc:
    print(s.subject)

# %%
youngest_group = sc.get_element("age-group-mean-18-24")
youngest_group.plot(
    regions=area3b_left, backend="plotly", min_connectivity=400, logscale=True
)

# %%
oldest_group = sc.get_element("age-group-mean-75-87")
oldest_group.plot(
    regions=area3b_left, backend="plotly", min_connectivity=400, logscale=True
)
