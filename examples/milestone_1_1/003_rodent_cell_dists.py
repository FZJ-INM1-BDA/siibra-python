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
Cell distributions in rodent brains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
import siibra
from nilearn import plotting
import matplotlib.pyplot as plt
import pandas as pd

# %%
# Cell distributions measurements are available for rodent and human brains.
# For mouse, features are typically anchored to regions of the Allen Mouse Brain Atlas.
amba = siibra.parcellations.get("Allen Mouse v3 2017")
features = siibra.features.get(amba, siibra.features.cellular.CellDistribution)
print(len(features), "cell distribution features anchored to Allen Mouse Brain Atlas.")

# %%
# More detail about the features is available via their metadata. For example,
# we can inspect the name, DOI and description of the corresponding datasets,
# to see that the distributions refer to paravalbumin positive neurons from the
# dataset by Kim et al. (2017).
# TODO Feature name is not descriptive enough
# TODO polish dictionary layout for URLs, confusing that the "urls" attribute has "url" as a key again.
dataset = features[0].datasets[0]
print(features[0].name)
print(dataset.description)

# %%
# The mouse cell distributions originate from the same dataset, but are anchored
# to different brain regions and stem from different subjects. Collecting the
# anchoring and subject information into a dataframe provides a nice overview of
# these several hundred variants.
pd.DataFrame([
    {"region": str(f.anchor), "subject": f.subject}
    for f in features
])

# %%
# Instead of using the whole atlas, the search query can directly target specific
# brain regions. Here we query all cell distributions anchored to subtree of
# olfactory areas, yielding about 40 variants.
olfactory = amba.get_region("Olfactory areas")
features = siibra.features.get(olfactory, "cell distribution")
featuretable = pd.DataFrame([
    {"region": str(f.anchor), "subject": f.subject}
    for f in features
])
featuretable

# %%
# We build a figure of the densities in the different olfactory subregions,
# plotted as points on the brain template per subject.
template = siibra.get_template("mouse").fetch()

# A merged olfactory segmentation mask is used to support the figure.
mask = siibra.volumes.volume.merge(
    [r.get_regional_mask("allen") for r in olfactory]
).fetch()

# For the plots, points will be color-grouped by the anchored brain area.
# TODO set all plots to teh same cut_coords for better comparison
for subject in featuretable.subject.unique():
    selection = [f for f in features if f.subject == subject]
    display = plotting.plot_roi(
        mask,
        bg_img=template,
        colorbar=False,
        draw_cross=False,
        title=subject,
        cut_coords=selection[0].data.mean(),
    )
    cmap = plt.get_cmap("tab10", len(selection))
    for i, cd in enumerate(selection):
        display.add_markers(cd.data, marker_color=cmap(i), marker_size=1)

# %%
# The same workflow can be applied to cell distributions in rat brains.
waxholm = siibra.parcellations.get("Waxholm rat (v4)")
features = siibra.features.get(waxholm, "cell distribution")
print(len(features), "cell distribution features anchored to Waxholm Rat Brain Atlas.")
dataset = features[0].datasets[0]
print(features[0].name)
print(dataset.description)

# %%
# Similarly, we can display the parvalbumin positive neurons in olfactory bulb
# for rat per subject
olfactory = waxholm.get_region("olfactory bulb")
mask = olfactory.get_regional_mask("waxholm").fetch()
features = siibra.features.get(olfactory, "cell distribution")
featuretable = pd.DataFrame([
    {"region": str(f.anchor), "subject": f.subject}
    for f in features
])
template = siibra.get_template("Waxholm").fetch()
for subject in featuretable.subject.unique():
    selection = [cd for cd in features if cd.subject == subject]
    display = plotting.plot_roi(
        mask,
        bg_img=template,
        colorbar=False,
        draw_cross=False,
        title=cd.subject,
        cut_coords=selection[0].data.mean(),
    )
    cmap = plt.get_cmap("jet", len(selection))
    for i, cd in enumerate(selection):
        display.add_markers(cd.data, marker_color=cmap(i), marker_size=1)
