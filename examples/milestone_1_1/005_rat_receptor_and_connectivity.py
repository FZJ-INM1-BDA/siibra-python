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
import matplotlib.pyplot as plt

# %%
injection_region = siibra.get_region(
    "waxholm rat", "Primary somatosensory area, forelimb representation"
)
tracing_conns = siibra.features.get(injection_region, "TracingConnectivityDistribution")
print(len(tracing_conns))
for f in tracing_conns:
    print(f.name)

# %%
print(tracing_conns[2].name)
print(tracing_conns[2].datasets[0].name)
print(tracing_conns[2].modality)
tracing_conns[2].data

# %%
waxholm_rat_template = siibra.get_template("waxholm").fetch(max_bytes=0.3 * 1024**3)
display = plotting.plot_img(
    img=waxholm_rat_template,
    bg_img=None,
    cmap="gray",
    title=f"red: {tracing_conns[2].name}\nblue: {tracing_conns[4].name}",
    cut_coords=tracing_conns[2].data.mean(axis=0),
)
display.add_markers(tracing_conns[2].data, marker_color="r", marker_size=1)
display.add_markers(tracing_conns[4].data, marker_color="b", marker_size=1)

# %%
# now let us query for receptor autoradiography sections coinciding with this
# region. Notice that the
autoradiography_sections = siibra.features.get(injection_region, "autoradiography")
for section in autoradiography_sections:
    print(section.name)

receptor_density_img = autoradiography_sections[0].fetch()

# %%
# Note that, the receptor autoradiography section is correctly anchored and
# siibra can find them according to their placement in the template, however,
# in this case nilearn.plotting falls short since it cannot handle
# non-orthogonal registrations.
cut_coords = injection_region.compute_centroids("waxholm")[0].coordinate
plotting.plot_roi(
    injection_region.get_regional_mask("waxholm").fetch(),
    bg_img=waxholm_rat_template,
    cut_coords=cut_coords,
)
plotting.plot_img(
    receptor_density_img,
    bg_img=waxholm_rat_template,
    cmap="magma",
    cut_coords=cut_coords,
)
# %%
# Therefore, we need to plot it with matplotlib
plt.imshow(receptor_density_img.dataobj.squeeze().T, aspect="equal")
