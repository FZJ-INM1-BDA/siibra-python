# Copyright 2018-2026
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
.. _011_streamline_fiber_bundles
:bdg-secondary:`Intermediate`

Streamline fiber bundles
~~~~~~~~~~~~~~~~~~~~~~~~

siibra provides tractograms of the Chenonceau brain, obtained with global
tractography, as streamline fiber bundles. Each tractogram is a TRX file
whose groups are the bundles. Querying a space returns the whole
tractograms, while querying a region or parcellation returns them
restricted to the bundles terminating in the query's leaf regions.
Reading TRX files requires `trx-python`, which siibra installs on
Python >= 3.11.
"""

# %%
import siibra
import numpy as np
import matplotlib.pyplot as plt
from siibra.features.connectivity import StreamlineFiberBundles
# sphinx_gallery_thumbnail_number = 1

# %%
# Whole tractograms are found by querying their space.
mni152 = siibra.get_space("mni152")
tractograms = siibra.features.get(mni152, StreamlineFiberBundles)
for tractogram in tractograms:
    print(tractogram.name)

# %%
# `bundles` lists the groups of the TRX file. For a whole tractogram, this
# loads the file, once per session. It is memory-mapped, so loading it does
# not read all streamlines into memory.
long_bundles = tractograms[0]
print(len(long_bundles.bundles), "bundles, e.g.", long_bundles.bundles[:5])

# %%
# `fetch` returns the requested bundles as a TRX file, with one group per
# bundle. Bundle names are matched exactly.
bundle = long_bundles.bundles[0]
trx = long_bundles.fetch([bundle])
print(list(trx.groups), "-", len(trx.streamlines), "streamlines")

# %%
# The streamlines are in the space of the feature, in RAS+ mm. They convert
# to a nibabel tractogram, e.g. for use with DIPY or nilearn.
tractogram = trx.to_tractogram()
print(tractogram.streamlines[0][:3])

# %%
# Querying a region returns each tractogram restricted to the bundles
# terminating in that region, anchored to it. These features know their
# bundles without loading the TRX file.
hoc1_left = siibra.get_region("julich 3.1", "hOc1 left")
for feature in siibra.features.get(hoc1_left, StreamlineFiberBundles):
    print(feature.name, "-", len(feature.bundles), "bundles")
    print("  anchored to:", feature.anchor.regions)

# %%
# For a region with subregions, we get the bundles terminating in any of its
# leaf regions. A parcellation is a region too: querying it returns the
# bundles terminating in any of its regions.
hoc1 = siibra.get_region("julich 3.1", "hOc1")
print("Leaves:", [leaf.name for leaf in hoc1.leaves])
for feature in siibra.features.get(hoc1, StreamlineFiberBundles):
    print(feature.name, "-", len(feature.bundles), "bundles")

# %%
# Fetching a filtered feature loads its bundles from the same memory-mapped
# file. We plot an axial projection of some streamlines of each bundle
# terminating in hOc1 left.
hoc1_left_bundles = siibra.features.get(hoc1_left, StreamlineFiberBundles)[0]
trx = hoc1_left_bundles.fetch()

fig, ax = plt.subplots(figsize=(7, 7))
rng = np.random.default_rng(0)
for i, (name, indices) in enumerate(trx.groups.items()):
    color = plt.cm.tab20(i % 20)
    for index in rng.choice(np.asarray(indices), size=min(50, len(indices)), replace=False):
        streamline = trx.streamlines[index]
        ax.plot(streamline[:, 0], streamline[:, 1], color=color, linewidth=0.3)
ax.set_aspect("equal")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title(f"Bundles terminating in {hoc1_left.name}")
plt.show()

# %%
# A selection can be saved as a TRX file for other tools with
# ``trx.trx_file_memmap.save(trx, "bundles.trx")``.
