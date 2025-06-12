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
)

# %%
print(fiber_bundles[0].bundle_id)


# %%
bundle_rh_0000000001 = fiber_bundles[0]
# regions this bundle passing through
print(bundle_rh_0000000001.regions)

# %%
# Each bundle is represented as a list of fibers which in turn are represented
# as Contour objects. Countours are just PointClouds where the order of the
# coordinates is important. (This enables warping the coordinates to other
# spaces effeciently).
print(type(bundle_rh_0000000001.data))
print(type(bundle_rh_0000000001.data[0]))

# %%
bundle_rh_0000000001.plot()

# %%
bigbrain = siibra.spaces["bigbrain"]
fiber_id = next(iter(bundle_rh_0000000001.fibers.keys()))
warped_fiber = bundle_rh_0000000001.fibers[fiber_id].warp(space=bigbrain)
display = plotting.plot_img(
    img=bigbrain.fetch(resolution_mm=1),
    bg_img=None,
    cmap="gray",
    title=f"Bundle: {bundle_rh_0000000001} - fiber: {fiber_id}",
)
display.add_markers(warped_fiber.coordinates)
