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
Mouse tracing connectivity distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
import siibra
from nilearn import plotting
import matplotlib.pyplot as plt

# %%
# The tracing connectivity features are anchored at the injection regions.
# To see all available options for mice, query for tracing connectivity
# distribution in the whole mouse atlas. We then get the published dataset names
# and descriptions to see what is the nature of the integrated data.
parcellation = siibra.parcellations["Allen Mouse v3 2017"]
all_rat_tracing_conn_dists = siibra.features.get(
    parcellation, "tracing connectivity distribution"
)
datasets = {
    "".join(ds.name for ds in tcd.datasets): f"{tcd.description}\nDOI: {''.join(tcd.urls)}"
    for tcd in all_rat_tracing_conn_dists
}
for name, desc in datasets.items():
    print("Name:", name)
    print("Description:", desc)
    print()

# %%
# Print the injection regions and subjects for each feature before making a
# finer region selection
for tcd in all_rat_tracing_conn_dists:
    print(f"Injection region: {tcd.anchor}, subject: {tcd.subject}")

# %%
# The dataset descriptions obtained above explain that there is one dataset for
# wild type mice and another for Cre-transgenic mice. Therefore, we can compare
# how the tracer travled in both after selecting a specific injection region;
# for example, anterior cingulate area, dorsal part.
injection_region = parcellation.get_region("Anterior cingulate area, dorsal part")
tcds_AntCinDorsal = siibra.features.get(
    injection_region,
    "tracing connectivity distribution"
)

# %%
# Furthermore, it is described that point data are derived from images of
# anterogradely labeled axonal projections from different cerebro-cortical
# locations to four subcortical brain regions. These regions are encoded in the
# subject name:
for f in tcds_AntCinDorsal:
    print(f.subject)

# %%
# Using these information, the tracing results can be compared between wild type
# and Cre-transgenic mice per subcortical region:
subcortical_regions = [
    "Caudoputamen",
    "PontineNuclei",
    "SuperiorColliculus",
    "Thalamus",
]
allen_mouse_template = siibra.get_template("mouse").fetch()
fig, axs = plt.subplots(len(subcortical_regions), 1, figsize=(15, 24))
for i, region in enumerate(subcortical_regions):
    tcd_cre = [
        tcd
        for tcd in tcds_AntCinDorsal
        if region in tcd.subject and "Cre-transgenic" in tcd.description
    ][0]
    tcd_wt = [
        tcd
        for tcd in tcds_AntCinDorsal
        if region in tcd.subject and "wild-type" in tcd.description
    ][0]
    display = plotting.plot_img(
        img=allen_mouse_template,
        bg_img=None,
        cmap="gray",
        title=region,
        cut_coords=tcd_wt.data.mean(axis=0),
        axes=axs[i],
        draw_cross=False,
        black_bg=True,
        colorbar=False,
    )
    display.add_markers(
        tcd_wt.data,
        marker_color="r",
        marker_size=1,
        label="Wild type"
    )
    display.add_markers(
        tcd_cre.data,
        marker_color="b",
        marker_size=1,
        label="Cre-transgenic",
    )
    plt.legend(loc="upper center", bbox_to_anchor=(1.4, 0.6), fontsize="x-large")
