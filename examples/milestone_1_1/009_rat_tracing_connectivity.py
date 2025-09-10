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
# The tracing connectivity features are anchored at the injection regions.
# To see all availabe options, we query for tracing connectivity distribution
# in the whole waxholm rat atlas.
parcellation = siibra.parcellations["waxholm rat"]
all_rat_tracing_conn_dists = siibra.features.get(
    parcellation, "tracing connectivity distribution"
)
for tcd in all_rat_tracing_conn_dists:
    print(f"Injection region: {tcd.anchor}, subject: {tcd.subject}")

# %%
# Using "primary motor area" as the injection region, query for features
# and display the metadata of the datasets.
injection_region = parcellation.get_region("primary motor area")
tracing_conn_dists = siibra.features.get(
    injection_region, "tracing connectivity distribution"
)
datasets = {
    "".join(
        ds.name for ds in tcd.datasets
    ): f"{tcd.description}\nDOI: {''.join(tcd.urls)}"
    for tcd in tracing_conn_dists
}
for name, desc in datasets.items():
    print("Name:", name)
    print("Description:", desc)
    print()

# %%
# The descriptions make it clear that the datasets differ in completeness of the
# projections, i.e., one from sensorimotor cortex and the other is the entirety
# of the cerebral cortex. Using this information, the results can be compared:
waxholm_rat_template = siibra.get_template("waxholm").fetch()
fig, axs = plt.subplots(len(tracing_conn_dists), 1, figsize=(15, 48))
for i, tcd in enumerate(tracing_conn_dists):
    color = "r" if "entire cerebral cortex" in tcd.description else "b"
    display = plotting.plot_img(
        img=waxholm_rat_template,
        bg_img=None,
        cmap="gray",
        title=f"subject: {tcd.subject}",
        cut_coords=tcd.data.mean(axis=0),
        axes=axs[i],
        draw_cross=False,
        black_bg=True,
    )
    display.add_markers(tcd.data, marker_color=color, marker_size=1)

# %%
# Additionally, these features contain extra information in the subject names,
# specifically the second part of the subject names corresponds to the tracer
# used. This information can be found in the detailed data descriptor found by
# following dois. So another comparsion can be made by based on the tracer of
# interests. As an example, filter out the features in which "Fr" tracer was
# used and display them on the waxholm rat template:
tcd_Fr = [conn for conn in tracing_conn_dists if conn.subject.split("_")[1] == "Fr"]

fig, axs = plt.subplots(len(tcd_Fr), 1, figsize=(15, 16))
for i, tcd in enumerate(tcd_Fr):
    display = plotting.plot_img(
        img=waxholm_rat_template,
        bg_img=None,
        cmap="gray",
        title=tcd.name,
        axes=axs[i],
        cut_coords=tcd.data.mean(axis=0),
    )
    display.add_markers(tcd.data, marker_color="r", marker_size=1)


# %%
# Similarly, the features using "BDA" tracer can be filtered out and displayed:
tcd_BDA = [
    conn for conn in tracing_conn_dists if conn.subject.split("_")[1] == "BDA"
]
fig, axs = plt.subplots(len(tcd_BDA), 1, figsize=(15, 16))
for i, tcd in enumerate(tcd_BDA):
    display = plotting.plot_img(
        img=waxholm_rat_template,
        bg_img=None,
        cmap="gray",
        title=tcd.name,
        axes=axs[i],
        cut_coords=tcd.data.mean(axis=0),
    )
    display.add_markers(tcd.data, marker_color="b", marker_size=1)
