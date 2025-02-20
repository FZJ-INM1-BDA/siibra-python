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
import pandas as pd
import re
import seaborn as sns
import siibra

assert siibra.__version__ >= "1.0.1"

sns.set_style("dark")

# %%
# # Instantiate parcellation and reference space from the human brain atlas

jubrain = siibra.parcellations.JULICH_BRAIN_CYTOARCHITECTONIC_ATLAS_V3_0_3
pmaps = jubrain.get_map(space="mni152", maptype="statistical")

# %%
# Obtain definitions and probabilistic maps in MNI asymmetric space of area IFG 44 and primary visual region hOc1 as defined in the Julich-Brain cytoarchitectonic atlas

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
    )
    plt.savefig(f"{r.key}.png")

# %%
# For each of the two brain areas, retrieve average densities of a selection of monogenetic neurotransmitter receptors, layer-specific distributions of cell bodies, as well as expressions of a selection of genes coding for these receptors.

receptors = ["M1", "M2", "M3", "D1", "5-HT1A", "5-HT2"]
genes = [
    siibra.vocabularies.GENE_NAMES.CHRM1,
    siibra.vocabularies.GENE_NAMES.CHRM2,
    siibra.vocabularies.GENE_NAMES.CHRM3,
    siibra.vocabularies.GENE_NAMES.HTR1A,
    siibra.vocabularies.GENE_NAMES.HTR2A,
    siibra.vocabularies.GENE_NAMES.DRD1,
]
modalities = [
    (siibra.features.molecular.ReceptorDensityFingerprint, {}, {"rot": 90}),
    (siibra.features.cellular.LayerwiseCellDensity, {}, {"rot": 0}),
    (siibra.features.molecular.GeneExpressions, {"gene": genes}, {"rot": 90}),
]
fig, axs = plt.subplots(
    len(modalities) + 1, len(regions), figsize=(4 * len(regions), 11), sharey="row"
)
ymax = [1000, 150, None]

for i, region in enumerate(regions):
    axs[0, i].imshow(plt.imread(f"{region.key}.png"))
    axs[0, i].set_title(f'{region.name.replace("Area ", "")}')
    axs[0, i].axis("off")
    for j, (modality, kwargs, plotargs) in enumerate(modalities):
        fts = siibra.features.get(region, modality, **kwargs)
        assert len(fts) > 0
        if len(fts) > 1:
            print(f"More than one feature found for {modality}, {region.name}")
        f = fts[0]
        if modality == siibra.features.molecular.ReceptorDensityFingerprint:
            fcopy = f
            fcopy._data_cached = f.data.loc[receptors]
        f.plot(ax=axs[j + 1, i], **plotargs)
        if modality == siibra.features.molecular.ReceptorDensityFingerprint:
            # add neurotransmitter names to  receptor names in xtick labels
            transmitters = [re.sub(r"(^.*\()|(\))", "", n) for n in f.neurotransmitters]
            axs[j + 1, i].set_xticklabels(
                [f"{r}\n({n})" for r, n in zip(f.receptors, transmitters)]
            )
        if ymax[j] is not None:
            axs[j + 1, i].set_ylim(0, ymax[j])
        if "std" in axs[j + 1, i].yaxis.get_label_text():
            axs[j + 1, i].set_ylabel(
                axs[j + 1, i].yaxis.get_label_text().replace("std", "std\n")
            )
        axs[j + 1, i].set_title(f"{fts[0].modality}")
fig.suptitle("")
fig.tight_layout()

# %%
# For each of the two brain areas, collect functional connectivity profiles referring to temporal correlation of fMRI timeseries of several hundred subjects from the Human Connectome Project. We show the strongest connections per brain area for the average connectivity patterns
fts = siibra.features.get(
    regions[0], siibra.features.connectivity.FunctionalConnectivity
)
conn = fts[0]

# aggregate connectivity profiles for first region across subjects
D1 = (
    pd.concat([c.get_profile(regions[0]).data for c in conn], axis=1)
    .agg(["mean", "std"], axis=1)
    .sort_values(by="mean", ascending=False)
)

# aggregate connectivity profiles for second region across subjects
D2 = (
    pd.concat([c.get_profile(regions[1]).data for c in conn], axis=1)
    .agg(["mean", "std"], axis=1)
    .sort_values(by="mean", ascending=False)
)


# plot both average connectivity profiles to target regions
def shorten_name(n):
    return (
        n.replace("Area ", "")
        .replace(" (GapMap)", "")
        .replace("left", "L")
        .replace("right", "R")
    )


fig, (a1, a2) = plt.subplots(1, 2, sharey=True, figsize=(3.6 * len(regions), 4.1))
kwargs = {"kind": "bar", "width": 0.85, "logy": True}
D1.iloc[:17]["mean"].plot(
    **kwargs, yerr=D1.iloc[:17]["std"], ax=a1, ylabel=shorten_name(regions[0].name)
)
D2.iloc[:17]["mean"].plot(
    **kwargs, yerr=D2.iloc[:17]["std"], ax=a2, ylabel=shorten_name(regions[1].name)
)
a1.set_ylabel("temporal correlation")
a1.xaxis.set_ticklabels(
    [shorten_name(t.get_text()) for t in a1.xaxis.get_majorticklabels()]
)
a2.xaxis.set_ticklabels(
    [shorten_name(t.get_text()) for t in a2.xaxis.get_majorticklabels()]
)
a1.grid(True)
a2.grid(True)
a1.set_title(regions[0].name.replace("Area ", ""))
a2.set_title(regions[1].name.replace("Area ", ""))
plt.suptitle("Functional Connectivity")
plt.tight_layout()

# %%
# For both brain areas, sample representative cortical image patches at 1µm
# resolution that were automatically extracted from scans of BigBrain sections.
# The image patches display clearly the differences in laminar structure of the two regions

selected_sections = [4950, 1345]
fig, axs = plt.subplots(1, len(regions))
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

plt.tight_layout()
