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
dMRI streamline bundles
~~~~~~~~~~~~~~~~~~~~~~~
"""
# %%
# We start by loading the library
import siibra
from nilearn import plotting

# %%
p = siibra.parcellations["julich 3.1"]
fiber_bundles = siibra.features.get(
    p, siibra.features.connectivity.StreamlineFiberBundle
)[0]

# %%
print(fiber_bundles[0].bundle_id)


# %%
bundle_rh_0000000001 = fiber_bundles[0]
# regions this bundle passing through
print(bundle_rh_0000000001.regions)

# %%
# Each bundle is represented as a dictionary of fibers which in turn are
# represented as Contour objects. Countours are just PointClouds where the order
# of the coordinates is important. (This enables warping the coordinates to
# other spaces effeciently).
bundle_rh_0000000001.fibers

# %%
# Alternatively, the fibers are stored as DataFrame for interoperability
bundle_rh_0000000001.data

# %%
bundle_rh_0000000001.plot()


# %%
fiber_id = 0
display = plotting.plot_img(
    img=siibra.get_template("mni152").fetch(resolution_mm=1),
    bg_img=None,
    cmap="gray",
    title=f"Bundle: {bundle_rh_0000000001} - fiber: {fiber_id}",
    cut_coords=bundle_rh_0000000001.fibers[fiber_id].coordinates[25],
)
display.add_markers(bundle_rh_0000000001.fibers[fiber_id].coordinates, marker_size=4)

# %%
warped_fiber = bundle_rh_0000000001.fibers[fiber_id].warp("bigbrain")
display = plotting.plot_img(
    img=siibra.get_template("bigbrain").fetch(resolution_mm=1),
    bg_img=None,
    cmap="gray",
    title=f"Bundle: {bundle_rh_0000000001} - fiber: {fiber_id}",
    cut_coords=warped_fiber.coordinates[25],
)
display.add_markers(warped_fiber.coordinates, marker_size=4)
