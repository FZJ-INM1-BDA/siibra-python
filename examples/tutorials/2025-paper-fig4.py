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
Multimodal comparison of two cortical brain areas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tutorial shows how maps and mutimodal regional measurements are obtained using siibra-python for language area IFG 44 and primary visual region hOc1, as defined in the Julich-Brain cytoarchitectonic atlas.
"""

# %%
import matplotlib.pyplot as plt
from nilearn import plotting
import re
import seaborn as sns
import siibra

assert siibra.__version__ >= "1.0.1"

sns.set_style("dark")

# %%
# Instantiate parcellation and reference space from the human brain atlas
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

jubrain = siibra.parcellations.JULICH_BRAIN_CYTOARCHITECTONIC_ATLAS_V3_0_3
pmaps = jubrain.get_map(space="mni152", maptype="statistical")
print(jubrain.publications[0]['citation'])

# %%
# Obtain definitions and probabilistic maps in MNI asymmetric space of area IFG 44
# and primary visual region hOc1 as defined in the Julich-Brain cytoarchitectonic atlas

specs = ["ifg 44 left", "hoc1 left"]
regions = [jubrain.get_region(spec) for spec in specs]
for r in regions:
    plotting.plot_glass_brain(
        pmaps.fetch(region=r),
        cmap="viridis",
        draw_cross=False,
        colorbar=False,
        annotate=False,
        symmetric_cbar=True,
        title=r.name,
    )

# %%
# For each of the two brain areas, query for layer-specific distributions of cell bodies.
fig, axs = plt.subplots(1, len(regions), sharey=True)
for i, region in enumerate(regions):
    layerwsise_cellbody_densities = siibra.features.get(region, "layerwise cell density")
    layerwsise_cellbody_densities[-1].plot(ax=axs[i])  # TODO: fix hoc1 and hoc2 issue
    print(layerwsise_cellbody_densities[-1].urls)
    axs[i].set_ylim(0, 150)

# %%
# Next, retrieve average densities of a selection of monogenetic
# neurotransmitter receptors.
receptors = ["M1", "M2", "M3", "D1", "5-HT1A", "5-HT2"]
fig, axs = plt.subplots(1, len(regions), sharey=True)
for i, region in enumerate(regions):
    receptor_fingerprints = siibra.features.get(region, "receptor density fingerprint")
    receptor_fingerprints[0].plot(receptors=receptors, ax=axs[i])
    print(receptor_fingerprints[0].urls)
    axs[i].set_ylim(0, 1000)
    transmitters = [re.sub(r"(^.*\()|(\))", "", n) for n in receptor_fingerprints[0].neurotransmitters]
    axs[i].set_xticklabels([f"{r}\n({n})" for r, n in zip(receptors, transmitters)])

# %%
# Now, for further insight, query for expressions of a selection of genes coding
# for these receptors.
genes = ["CHRM1", "CHRM2", "CHRM3", "HTR1A", "HTR2A", "DRD1"]
fig, axs = plt.subplots(1, len(regions), sharey=True)
for i, region in enumerate(regions):
    gene_expressions = siibra.features.get(region, "gene expressions", gene=genes)
    assert len(gene_expressions) == 1
    gene_expressions[0].plot(ax=axs[i])
    print(gene_expressions[0].urls)

# %%
# For each of the two brain areas, collect functional connectivity profiles referring to
# temporal correlation of fMRI timeseries of several hundred subjects from the Human Connectome
# Project. We show the strongest connections per brain area for the average connectivity patterns
fts = siibra.features.get(jubrain, "functional connectivity")
for cf in fts:
    if cf.cohort != "HCP":
        continue
    if cf.paradigm == "Resting state (EmpCorrFC concatenated)":
        conn = cf
        break
print(conn.urls)  # TODO: add publication to conn json


# plot both average connectivity profiles to target regions
def shorten_name(n):
    return (
        n.replace("Area ", "")
        .replace(" (GapMap)", "")
        .replace("left", "L")
        .replace("right", "R")
    )


plotkwargs = {
    "kind": "bar",
    "width": 0.85,
    "logscale": True,
    "xlabel": "",
    "ylabel": "temporal correlation",
    "rot": 90,
}
fig, axs = plt.subplots(1, len(regions), sharey=True)
for i, region in enumerate(regions):
    plotkwargs["ax"] = axs[i]
    conn.plot(region, max_rows=17, **plotkwargs)
    axs[i].xaxis.set_ticklabels([shorten_name(t.get_text()) for t in axs[i].xaxis.get_majorticklabels()])
    axs[i].set_title(region.name.replace("Area ", ""))
    axs[i].grid(True, 'minor')
plt.suptitle("Functional Connectivity")

# %%
# For both brain areas, sample representative cortical image patches at 1µm
# resolution that were automatically extracted from scans of BigBrain sections.
# The image patches display clearly the differences in laminar structure of the two regions

selected_sections = [4950, 1345]
fig, axs = plt.subplots(1, len(regions), sharey=True)
for r, sn, ax in zip(regions, selected_sections, axs):
    # find 1 micron sections intersecting the region
    pmap = pmaps.get_volume(r)
    all_patches = siibra.features.get(pmap, "BigBrain1MicronPatch")
    patches = [p for p in all_patches if p.bigbrain_section == sn]

    # select the first patch and access the underlying image data
    patchimg = patches[0].fetch()
    patchdata = patchimg.get_fdata().squeeze()

    # plot the pure image array
    ax.imshow(patchdata, cmap="gray", vmin=0, vmax=2**16)
    ax.axis("off")
    ax.set_title(r.name)

# sphinx_gallery_thumbnail_number = 2
