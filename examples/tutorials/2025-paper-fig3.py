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
optional certainty quantifiers.
"""

# %%
import siibra
assert siibra.__version__ >= "1.0.1"
from nilearn import plotting, image
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style('dark')

# %%
# # Input: Activation map or other feature distribution as image volume in MNI space
#
# We compose an artificial input image by merging some functional maps from the DiFuMo atlas, but the image could be any feature distribution map. The image is built as a NIfTI, but then casted to a siibra volume so we have a reference space attached and can used it properly in the siibra workflows. The NIfTI object can always be accessed via `.fetch()`. 

# obtain and merge a couple of functional statistical maps
difumo128 = siibra.get_map(parcellation="difumo 64", space='mni152', maptype='statistical')
img = image.smooth_img(image.math_img(
    "np.maximum(np.maximum(im1, im2), im3)", 
    im1=difumo128.fetch(region="3"), 
    im2=difumo128.fetch(region="31"),
    im3=difumo128.fetch(region="30"), 
), 10)

# Embed the NIfTI image as a siibra volume with space attached.
input_volume = siibra.volumes.from_nifti(img, space='mni152', name='example input volume')
plotting.plot_glass_brain(input_volume.fetch(), alpha=1, cmap='RdBu', symmetric_cbar=True)

# %%
# # Split input volume into cluster components
#
# There are many ways to get components out of a feature map. Here we use siibra to 
# - draw random points from the distribution encoded by the input volume, then 
#  cluster them using DBSCAN, and 
# - build clusterwise featuremaps as Kernel Density estimates thereof.
#
# In this example, this more or less inverts the composition of the input volume from the DiFuMo maps, but the idea for a general input image is to separate it into components that have more meaningful correlations with brain regions than the full image, which is usually a mixture distribution. 

np.random.seed(25)
N = 10000  # number of random samples
# drawing the samples results in a siibra PointCloud, 
# which has reference space attached and can model point uncertainties.
samples = input_volume.draw_samples(N, e=5, sigma_mm=3)

# finding the clusters will result in a labelling of the point set.
samples.find_clusters(min_fraction=1/300, max_fraction=1/2)
clusterlabels = set(samples.labels) - {-1}

# Let's have a look at the clustered pointcloud
view = plotting.plot_glass_brain(input_volume.fetch(), alpha=1, threshold=15, cmap='RdGy')
view.add_markers(
    np.array(samples.as_list())[samples.labels >= 0],
    marker_size=5,
    marker_color=[samples.label_colors[l] for l in samples.labels if l >= 0]
)

# %%
# # Assign peaks and clusters to cytoarchitectonic regions
#
# To assign the clusters to brain regions, we build feature maps from each cluster and assign them to the Julich-Brain probabilistic maps. The assignemint is one-to-many since the structures in the image and parcellation are continuous. Assignments report correlation, intersection over union, and some other measures which we can use to filter and sort them.
#
# The result is an assignment table from cluster components in the input volume to regions in the Julich-Brain atlas.

min_correlation = 0.2
min_map_value = 0.5
pmaps = siibra.get_map(parcellation="julich 3.1", space="mni152", maptype='statistical', spec='207')
assignments = []

# assign peaks to regions
peaks = input_volume.find_peaks(mindist=5, sigma_mm=0)
with siibra.QUIET:
    df = pmaps.assign(peaks)
df.query(f"`map value` >= {min_map_value}", engine='python', inplace=True)
df['type'] = 'peak'
df['id'] = df['input structure']   
assignments.append(df[['type', 'id', 'region', 'map value']])

view = plotting.plot_glass_brain(input_volume.fetch(), alpha=1, cmap='RdBu', symmetric_cbar=True)
view.add_markers(peaks.as_list(), marker_size=30)

# %%  
# assign clusters to regions
for l in clusterlabels:
    clustermap = siibra.volumes.from_pointcloud(samples, label=l, target=input_volume)
    plotting.plot_glass_brain(
        clustermap.fetch(), alpha=1, cmap='RdBu', title=f"Cluster #{l}",
        symmetric_cbar=True
    )
    with siibra.QUIET:
        df = pmaps.assign(clustermap)
    df.query(f"correlation >= {min_correlation}", engine='python', inplace=True)
    df['type'] = 'cluster'
    df['id'] = l    
    assignments.append(df[['type', 'id', 'region', 'correlation', 'map value']])

all_assignments = pd.concat(assignments).sort_values(by='correlation', ascending=False)
all_assignments

# %%
# plot the three primary assigned probability maps
regions = set()
for n, a in all_assignments.iterrows():
    if a.region in regions:
        continue
    pmap = pmaps.fetch(a.region)
    plotting.plot_glass_brain(pmap, cmap='hot_r')
    regions.add(a.region)
    print(a.region, a.correlation)
    if len(regions) == 3:
        break

# %%
# # Find features
# To demonstrate multimodal feature profiling, we only choose the first connected region.

def shortname(region):
    return (
        re.sub('\s*\(.*\)\s*|\s*Areaa\s*', ' ', region.name)
        .replace('left', 'L')
        .replace('right', 'R')
        .strip()
    )

def filter_features(feats, region):
    return [f for f in feats if any(r.assign(region) for r in f.anchor.regions)]

def plot_receptors(region, ax):
    fts = filter_features(
        siibra.features.get(region, siibra.features.molecular.ReceptorDensityFingerprint),
        region
    )
    fts[0].plot(ax=ax)

def plot_celldensities(region, ax):
    fts = filter_features(
        siibra.features.get(region, siibra.features.cellular.LayerwiseCellDensity),
        region
    )
    fts[0].plot(ax=ax)

def plot_gene_expressions(region, ax, genes):
    fts = siibra.features.get(region, siibra.features.molecular.GeneExpressions, gene=genes)
    fts[0].plot(ax=ax)

def plot_connectivity(region, ax):
    fts = siibra.features.get(region, siibra.features.connectivity.StreamlineCounts)
    conndata = fts[0][0].get_profile(region).data
    conndata.rename(index={r:shortname(r) for r in conndata.index}, inplace=True)
    conndata[:15].plot(kind='bar', ax=ax)
    plt.xticks(ha='right')
    plt.tight_layout()
    plt.grid(True)
    plt.title(f'Connectivity for {region.name}')


# %%
selected_region = siibra.get_region("julich 3.1", "Area hOc1 (V1, 17, CalcS) left")
plot_funcs = [
    lambda r, a: plot_receptors(r, a),
    lambda r, a: plot_celldensities(r, a),
    lambda r, a: plot_connectivity(r, a),
    lambda r, a: plot_gene_expressions(r, a, ['gabarapl1', 'gabarapl2', 'maoa', 'tac1']),
]

fig, axs = plt.subplots(1, len(plot_funcs), figsize=(3.5 * len(plot_funcs), 4))
for ax, func in zip(axs.ravel(), plot_funcs):
    func(r=selected_region, a=ax)
    ax.grid(True)
    xtl = ax.get_xticklabels()
    ax.set_xticklabels(xtl, rotation=55, ha='right')
plt.suptitle('')
plt.tight_layout()