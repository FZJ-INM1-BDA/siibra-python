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

Anatomical characterization and multimodal profiling of regions of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`siibra` accepts user-defined location specifications in the form of reference space coordinates with
optional certainty quantifiers, or feature maps from imaging experiments.
Feature maps are typically separated into cluster components to obtain clearly localized regions of interest (ROIs).
Any region of interest in image form can be used to run spatial feature queries and extract multimodal data features co-localized with ROIs.
In addition, locations of interest are assigned to brain areas using probabilistic maps from functional, cyto- or fiber architectonic reference atlases, distinguishing incidence, correlation and overlap of structures. The resulting associated brain areas
reveal additional multimodal data features characterizing the ROIs.
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting, image
import seaborn as sns
import siibra

assert siibra.__version__ >= "1.0.1"

sns.set_style("dark")


# %%
# Input: Activation map or other feature distribution as image volume in MNI space
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We compose an artificial input image by merging some functional maps from the DiFuMo atlas,
# but the image could be any feature distribution map. The image is built as a NIfTI, but
# then casted to a siibra volume so we have a reference space attached and can used it
# properly in the siibra workflows. The NIfTI object can always be accessed via `.fetch()`.

# obtain and merge a couple of functional statistical maps
difumo128 = siibra.get_map(
    parcellation="difumo 64", space="mni152", maptype="statistical"
)
img = image.smooth_img(
    image.math_img(
        "np.maximum(np.maximum(im1, im2), im3)",
        im1=difumo128.fetch(region="3"),
        im2=difumo128.fetch(region="31"),
        im3=difumo128.fetch(region="30"),
    ),
    10,
)

# Embed the NIfTI image as a siibra volume with space attached.
input_volume = siibra.volumes.from_nifti(
    img, space="mni152", name="example input volume"
)
plotting.plot_glass_brain(
    input_volume.fetch(), alpha=1, cmap="RdBu", symmetric_cbar=True
)

# %%
# Split input volume into cluster components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# There are many ways to get components out of a feature map. Here we use siibra to
# - draw random points from the distribution encoded by the input volume, then
#  cluster them using DBSCAN, and
# - build clusterwise featuremaps as Kernel Density estimates thereof.
#
# In this example, this more or less inverts the composition of the input volume from
# the DiFuMo maps, but the idea for a general input image is to separate it into
# components that have more meaningful correlations with brain regions than the full
# image, which is usually a mixture distribution.

np.random.seed(25)
N = 10000  # number of random samples
# drawing the samples results in a siibra PointCloud,
# which has reference space attached and can model point uncertainties.
samples = input_volume.draw_samples(N, e=5, sigma_mm=3)

# finding the clusters will result in a labelling of the point set.
samples.find_clusters(min_fraction=1 / 300, max_fraction=1 / 2)
clusterlabels = set(samples.labels) - {-1}

# Let's have a look at the clustered pointcloud
view = plotting.plot_glass_brain(
    input_volume.fetch(), alpha=1, threshold=15, cmap="RdGy"
)
view.add_markers(
    np.array(samples.as_list())[samples.labels >= 0],
    marker_size=5,
    marker_color=[samples.label_colors[lb] for lb in samples.labels if lb >= 0],
)

# %%
# Assign clusters to cytoarchitectonic regions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To assign the clusters to brain regions, we build feature maps from each cluster
# and assign them to the Julich-Brain probabilistic maps. The assignment is one-to-many
# since the structures in the image and parcellation are continuous. Assignments
# report correlation, intersection over union, and some other measures which we can use
# to filter and sort them. The result is an assignment table from cluster components
# in the input volume to regions in the Julich-Brain atlas.

min_correlation = 0.2
min_map_value = 0.5
pmaps = siibra.get_map(
    parcellation="julich 3.0.3", space="mni152", maptype="statistical"
)

assignments = []
for cl in clusterlabels:
    clustermap = siibra.volumes.from_pointcloud(samples, label=cl, target=input_volume)
    plotting.plot_glass_brain(
        clustermap.fetch(),
        alpha=1,
        cmap="RdBu",
        title=f"Cluster #{cl}",
        symmetric_cbar=True,
    )
    with siibra.QUIET:
        df = pmaps.assign(clustermap)
    df.query(f"correlation >= {min_correlation}", engine="python", inplace=True)
    df['id'] = cl
    assignments.append(df[["id", "region", "correlation", "map value"]])

all_assignments = pd.concat(assignments).sort_values(by="correlation", ascending=False)
all_assignments[:9].round(2)

# %%
# plot the three primary assigned probability maps
regions = set()
for n, a in all_assignments.iterrows():
    if a.region in regions:
        continue
    pmap = pmaps.fetch(a.region)
    plotting.plot_glass_brain(pmap, cmap="hot_r")
    regions.add(a.region)
    print(a.region, a.correlation)
    if len(regions) == 3:
        break

# %%
# Find features
# ^^^^^^^^^^^^^
#
# To demonstrate multimodal feature profiling, we choose one of the connected regions.
selected_region = siibra.get_region("julich 3.0.3", "Area hOc1 (V1, 17, CalcS) left")

# %%
# Let us first query receptor density fingerprint for this region
receptor_fingerprints = siibra.features.get(selected_region, "receptor density fingerprint")
fp = receptor_fingerprints[0]
print("\n".join(fp.urls))
fig, axs = plt.subplots(1, 1, figsize=(4, 3.2))
fp.plot(ax=axs, capsize=4)


# %%
# Now, query for gene expressions for the same region
genes = ["gabarapl1", "gabarapl2", "maoa", "tac1"]
gene_expressions = siibra.features.get(selected_region, "gene expressions", gene=genes)
print("\n".join(gene_expressions[0].urls))
fig, axs = plt.subplots(1, 1, figsize=(4, 3.5))
gene_expressions[0].plot(ax=axs)


# %%
# We can next obtain cell body density values for this region
cell_densities = siibra.features.get(selected_region, "LayerwiseCellDensity")
print("\n".join(cell_densities[0].urls))
fig, axs = plt.subplots(1, 1, figsize=(4, 3.2))
cell_densities[0].plot(ax=axs)

# %%
# Lastly, we can obtain the regional profile of streamline count type
# parcellation-based connectivity matrices
connectivity_profiles = siibra.features.get(selected_region, "StreamlineCounts")
conn = connectivity_profiles[0]
print("\n".join(conn.urls))


def shorten_name(region):
    # to simplify readability
    return (
        region.replace("Area ", "")
        .replace(" (GapMap)", "")
        .replace("left", "L")
        .replace("right", "R")
    )


fig, axs = plt.subplots(1, 1, figsize=(4, 4.5))
conn.plot(selected_region, ax=axs, max_rows=15, kind="bar", rot=50, width=0.7)
axs.set_ylabel(
    f"Mean of {conn.modality} \u00b1 std in {len(conn.elements)} {conn.indexing_attributes[0]}s",
    wrap=True)
axs.xaxis.set_ticklabels([shorten_name(t.get_text()) for t in axs.xaxis.get_majorticklabels()])
plt.grid(True, 'minor')
plt.title(f"Connectivity for {selected_region.name}")
