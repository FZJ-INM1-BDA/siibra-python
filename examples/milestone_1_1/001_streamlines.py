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
dMRI streamline matrices and fiber bundles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# %%
# We start by loading the library
import siibra
from siibra.volumes.volume import ReducedVolume
from nilearn import plotting

# %%
# We choose a cortical region from Julich Brain and find fiber bundles
# overlapping with this region
julich_brain = siibra.parcellations["julich 3.1"]
area3b_left = julich_brain.get_region("Area 3b (PostCG) left")
bundles_passing_3bleft = siibra.features.get(
    area3b_left, siibra.features.connectivity.StreamlineFiberBundle
)
print("Bundles found:", len(bundles_passing_3bleft))
print(bundles_passing_3bleft[12].name)
print(bundles_passing_3bleft[12].description)


# %%
# Each bundle is represented as a dictionary of fibers which in turn are
# represented as Contour objects. Countours are just PointClouds where the order
# of the coordinates is important. (This enables warping the coordinates to
# other spaces effeciently). Let us choose a bundle to demonstrate
bundle = bundles_passing_3bleft[12]
bundle.data


# %%
print("Fiber count:", len(bundle.data.index.unique()))
plotting.plot_markers(
    node_values=bundle.data.index,
    node_coords=bundle.data.values,
    node_cmap="jet",
    colorbar=False,
)


# %%
fibers = bundle.get_fibers()
print(type(fibers[0]))

# %%
fiber_id = 0
display = plotting.plot_img(
    img=siibra.get_template("mni152").fetch(resolution_mm=1),
    bg_img=None,
    cmap="gray",
    title=f"Bundle: {bundle.name} - fiber: {fiber_id}",
    cut_coords=fibers[fiber_id].coordinates[25],
)
display.add_markers(fibers[fiber_id].coordinates, marker_size=2)

# %%
warped_fiber = fibers[fiber_id].warp("bigbrain")
display = plotting.plot_img(
    img=siibra.get_template("bigbrain").fetch(resolution_mm=1),
    bg_img=None,
    cmap="gray",
    title=f"Bundle: {bundle.name} - fiber: {fiber_id}",
    cut_coords=warped_fiber.coordinates[25],
)
display.add_markers(warped_fiber.coordinates, marker_size=2)


# %%
features = siibra.features.get(julich_brain, "streamlinecounts")
for f in features:
    print(f.cohort)
# %%
cf = list(filter(lambda f: f.cohort == "1000BRAINS", features))[0]
print(cf.name)

# %%
for f in cf:
    print(f.subject)

# %%
cf[0].plot(regions=area3b_left, backend="plotly", min_connectivity=400)

# %%
profile = cf[0].get_profile(region=area3b_left, min_connectivity=400).data
profile.head()

# %%
area_1_left = profile.index[0]
bundles_passing_both = [
    f for f in bundles_passing_3bleft if area_1_left in f.anchor.regions
]
for bundle in bundles_passing_both:
    print(bundle.name)

# %%
mp = julich_brain.get_map("mni152")
img = ReducedVolume(
    [mp.get_volume(r) for r in [area_1_left, area3b_left]],
    [mp.get_index(r).label for r in [area_1_left, area3b_left]],
).fetch()
display = plotting.plot_glass_brain(img, title=f"Bundle: {bundle.name}", cmap="Blues")
display.add_markers(
    bundle.data.values, marker_size=1, marker_color=bundle.data.index, cmap="jet"
)

# %%
# Now, plot the masks of the regions overlapping with it
regionnames = {
    r.name for r in bundle.anchor.regions
}  # regions this bundle passing through
print(regionnames)

img_regions = ReducedVolume(
    [mp.get_volume(r) for r in regionnames],
    [mp.get_index(r).label for r in regionnames],
).fetch()
plotting.view_img(
    img_regions,
    symmetric_cmap=False,
    colorbar=False,
    cmap=julich_brain.get_map("mni152").get_colormap(regionnames),
)
