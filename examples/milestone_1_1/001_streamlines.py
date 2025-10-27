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
dMRI streamline fiber bundles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# %%
import siibra
from nilearn import plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import numpy as np

# %%
# Fiber bundles are anchored by assigning their terminal points to brain areas in supported reference atlases.
# This allows siibra to query bundles by brain area.
# Here, we query bundles terminating in area 3b (PostCG) in the left hemisphere.
julich_brain = siibra.parcellations.get("julich 3.1")
area3b_left = julich_brain.get_region("Area 3b (PostCG) left")
bundles_3bleft = siibra.features.get(
    area3b_left, siibra.features.connectivity.StreamlineFiberBundle
)
print(f"Found {len(bundles_3bleft)} bundles terminating in {area3b_left.name}.")


# %%
# The modality attribute can be used to filter only the short bundles.
# We use this here to select short superficial bundle terminating in area 3b left.
# The bundle is represented as a pandas DataFrame, containing x/y/z coordinates of streamlines in MNI space.
# Streamlines are concatenated and indexed by their fiber id.
# Coordinates of each streamline are ordered.
# TODO Update modality descriptions, one would expect "Superficial" and "deep white matter" 
short_bundles = [b for b in bundles_3bleft if "short" in b.modality]
bundle = short_bundles[0]
print("Fiber count:", len(bundle.data.index.unique()))
bundle.data


# %%
# The coordinates of streamlines can be plotted using nilearn.
# We color them by fiber id to distinguish different streamlines.
# Overlay terminal regions of the bundle are listed in the anchor attribute
# of the data feature. We plot these as contours for reference.
bundle_plot_settings = {
    "node_cmap": "turbo",
    "node_size": 0.5,
    "colorbar": False,
    "alpha": 0.3,
    "node_kwargs": {"marker": "o", "linewidths": 0}
}

fig = plt.figure(figsize=(10, 4), dpi=300)
display = plotting.plot_markers(
    bundle.data.index,
    bundle.data.values,
    figure=fig,
    **bundle_plot_settings,
)
display.title(bundle.name.replace('_', ' '), size=12)

# A reusable function to plot region contours with a legend
def add_region_contours(regions, display, linewidth=1):
    colors = plt.get_cmap("tab10")(range(len(regions)))
    for c, r in zip(colors, regions):
        display.add_contours(
            r.get_regional_map("mni152").fetch(),
            colors=[c], linewidths=linewidth
        )
    patches = [
        mpatches.Patch(color=c, label=re.sub(r"\([^)]*\)", "", r.name))
        for c, r in zip(colors, regions)
    ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

add_region_contours(bundle.anchor.regions, display, linewidth=1.)


# %%
# Long bundles can be selected in the same fashion.
# We plot the left CST bundle and its terminal regions in separate panels.

# find the bundle
long_bundles = [b for b in bundles_3bleft if "long" in b.modality]
for bundle in long_bundles:
    if "Left_CST" in bundle.name:
        break

# plot the streamlines in the bundle
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300)
display = plotting.plot_markers(
    bundle.data.index,
    bundle.data.values,
    axes=ax1,
    **bundle_plot_settings
)
display.title(bundle.name.replace('_', ' '), size=12)

# Define names and colors for the terminal regions
regions = list(bundle.anchor.regions.keys())
display = plotting.plot_glass_brain(None, axes=ax2, **bundle_plot_settings)
display.title('Terminal regions', size=12)
add_region_contours(regions, display)


# %%
