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
Rat tracing connectivity distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


# %%
import siibra
from nilearn import plotting
import matplotlib.pyplot as plt


# %%
# Tracing connectivity features in the rat brain are anchored at the injection regions.
# To see all available options, we query for tracing connectivity distribution
# in the whole waxholm rat atlas.
waxholm_parc = siibra.parcellations.get("waxholm rat")
features_tracing = siibra.features.get(
    waxholm_parc, "tracing connectivity distribution"
)
feature_table = siibra.features.tabulate(
    features_tracing,
    ["anchor", "subject", "name", "datasets"],
    converters={"anchor": str, "datasets": lambda ds: next(iter(ds))},
)

# %%
# A detailed description is included in the dataset attribute of each feature.
# Let's pick one injection region and inspect the available dataset descriptions
selection = feature_table.query("anchor == 'primary motor area'")

# selected features originate from the same dataset
datasets = selection.datasets.unique()
print(f"{len(datasets)} of unique datasets in selection.")
print(datasets[0].name)
print(datasets[0].description)

# %%
# The tabular data in each feature contains x/y/z coordinates of tracer
# projections originating from the injection region. We visualize the points 
# for different subjets and tracers of the selected injection brain region.
template = siibra.get_template("waxholm").fetch()
fig, axs = plt.subplots(len(selection), 1, figsize=(10, 27))
for i, tcd in enumerate(selection.feature):
    color = "r" if "cortex" in tcd.description else "b"
    s, t = tcd.subject.split("_")
    display = plotting.plot_img(
        img=template,
        bg_img=None,
        cmap="gray",
        title=f"subject: {s}, tracer: {t}",
        cut_coords=tcd.data.mean(axis=0),
        axes=axs[i],
        draw_cross=False,
        black_bg=True,
    )
    display.add_markers(tcd.data, marker_color=color, marker_size=1)

