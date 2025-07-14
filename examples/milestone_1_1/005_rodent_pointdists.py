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
Rodent tracing connectivity distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
import siibra
from nilearn import plotting

# %%
region = siibra.get_region(
    "waxholm rat", "Primary somatosensory area, forelimb representation"
)
tracing_conns = siibra.features.get(region, "TracingConnectivityDistribution")
print(len(tracing_conns))
for f in tracing_conns:
    print(f.name)

# %%
print(tracing_conns[2].name)
print(tracing_conns[2].datasets[0].name)
print(tracing_conns[2].modality)
tracing_conns[2].data

# %%
display = plotting.plot_img(
    img=siibra.get_template("waxholm").fetch(),
    bg_img=None,
    cmap="gray",
    title=f"red: {tracing_conns[2].name}\nblue: {tracing_conns[4].name}",
    cut_coords=tracing_conns[2].data.mean(axis=0),
)
display.add_markers(tracing_conns[2].data, marker_color="r", marker_size=1)
display.add_markers(tracing_conns[4].data, marker_color="b", marker_size=1)


# %%
region = siibra.get_region("mouse", "Supplemental somatosensory area")
tracing_conns = siibra.features.get(region, "TracingConnectivityDistribution")
print(len(tracing_conns))
for f in tracing_conns:
    print(f.name)

# %%
print(tracing_conns[3].name)
print(tracing_conns[3].datasets[0].name)
print(tracing_conns[3].modality)
print(tracing_conns[3].description)
tracing_conns[3].data

# %%
display = plotting.plot_img(
    img=siibra.get_template("mouse").fetch(),
    bg_img=None,
    cmap="gray",
    title=f"red: {tracing_conns[3].name}\nblue: {tracing_conns[7].name}",
    cut_coords=tracing_conns[3].data.mean(axis=0),
)
display.add_markers(tracing_conns[3].data, marker_color="r", marker_size=1)
display.add_markers(tracing_conns[7].data, marker_color="b", marker_size=1)
