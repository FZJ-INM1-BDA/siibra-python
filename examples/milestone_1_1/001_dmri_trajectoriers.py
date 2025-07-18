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
from siibra.volumes.volume import ReducedVolume
from nilearn import plotting


p = siibra.parcellations["julich 2.9"]
area3b_left = p.get_region("Area 3b (PostCG) left")
bundles_passing_3bleft = siibra.features.get(
    area3b_left, siibra.features.connectivity.StreamlineFiberBundle
)
print(len(bundles_passing_3bleft))
print(bundles_passing_3bleft[0].name)
print(bundles_passing_3bleft[0].modality)
print(bundles_passing_3bleft[0].description)

# %%
bundle = bundles_passing_3bleft[0]
regionnames = {
    r.name for r in bundle.anchor.regions
}  # regions this bundle passing through
print(regionnames)
mp = p.get_map("mni152")
img_regions = ReducedVolume(
    [mp.get_volume(r) for r in regionnames],
    [mp.get_index(r).label for r in regionnames],
).fetch()
plotting.view_img(
    img_regions,
    symmetric_cmap=False,
    colorbar=False,
    cmap=p.get_map("mni152").get_colormap(regionnames),
)

# %%
# Each bundle is represented as a dictionary of fibers which in turn are
# represented as Contour objects. Countours are just PointClouds where the order
# of the coordinates is important. (This enables warping the coordinates to
# other spaces effeciently).
fibers = bundle.get_fibers()

# %%
# Alternatively, the fibers are stored as DataFrame for interoperability
bundle.data

# %%
fiber_id = 0
display = plotting.plot_img(
    img=siibra.get_template("mni152").fetch(resolution_mm=1),
    bg_img=None,
    cmap="gray",
    title=f"Bundle: {bundle} - fiber: {fiber_id}",
    cut_coords=fibers[fiber_id].coordinates[25],
)
display.add_markers(fibers[fiber_id].coordinates, marker_size=2)

# %%
warped_fiber = fibers[fiber_id].warp("bigbrain")
display = plotting.plot_img(
    img=siibra.get_template("bigbrain").fetch(resolution_mm=1),
    bg_img=None,
    cmap="gray",
    title=f"Bundle: {bundle} - fiber: {fiber_id}",
    cut_coords=warped_fiber.coordinates[25],
)
display.add_markers(warped_fiber.coordinates, marker_size=2)

# %%
area44_left = p.get_region("Area 44 left")
bundles_passing_both = [
    f for f in bundles_passing_3bleft if area44_left in f.anchor.regions
]
for bundle in bundles_passing_both:
    print(bundle.name)

# %%
img = ReducedVolume(
    [mp.get_volume(r) for r in [area44_left, area3b_left]],
    [mp.get_index(r).label for r in [area44_left, area3b_left]],
).fetch()
display = plotting.plot_glass_brain(
    img,
    title=f"Bundle: {bundle.name}",
    cmap=mp.get_colormap([area44_left, area3b_left])
)
display.add_markers(bundle.data.values, marker_size=1)


# %%
display = plotting.plot_glass_brain(
    img,
    title=f"Bundle: {bundle.name}",
    cmap=mp.get_colormap([area44_left, area3b_left])
)
display.add_markers(bundle.data.values, marker_size=1, marker_color=bundle.data.index)
# %%
