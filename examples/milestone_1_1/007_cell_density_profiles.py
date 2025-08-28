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
Cell density Profiles
~~~~~~~~~~~~~~~~~~~~~
"""

# %%
import siibra
from nilearn import plotting

# %%
region = siibra.get_region("julich 3.1", "hoc6")
profiles = siibra.features.get(region, "cell density profile")
for pf in profiles:
    print(pf.name)

# %%
pf.data

# %%
pf.plot(y="density_mean", error_y="density_std", backend="plotly", kind="line")

# %%
layerwise = siibra.features.get(region, "layer cell density v2")
for lf in layerwise:
    print(lf.name)

# %%
lf.data

# %%
lf.plot(y="cell_size_mean_um2", error_y="cell_size_std_um2", backend="plotly")

# %%
cell_dists = siibra.features.get(region, siibra.features.cellular.CellDistribution)
for cd in cell_dists:
    print(cd.name)
ptcld = cd.as_pointcloud()

# %%
mni152_tmpl_img_voi = siibra.get_template("bigbrain").fetch(voi=ptcld.boundingbox)
display = plotting.plot_img(
    img=mni152_tmpl_img_voi,
    cut_coords=ptcld.centroid.coordinate,
    cmap="gray",
    display_mode="y",
)
display.add_markers(ptcld.coordinates, marker_size=1)

# %%
mni152_tmpl_img_voi = siibra.get_template("mni152").fetch()
ptcld_warped = ptcld.warp('mni152')
display = plotting.plot_img(
    img=mni152_tmpl_img_voi,
    cut_coords=ptcld_warped.centroid.coordinate,
    cmap="gray",
    display_mode="y",
)
display.add_markers(ptcld_warped.coordinates, marker_size=1)

