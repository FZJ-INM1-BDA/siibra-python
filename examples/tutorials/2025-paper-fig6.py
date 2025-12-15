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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Case study: Anatomical evaluation of subcortical maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This notebook uses siibra to re-assess some results of the study "Ventral intermediate nucleus structural connectivity-derived segmentation: anatomical reliability and variability", [Bertino et al., NeuroImage 2021](https://doi.org/10.1016/j.neuroimage.2021.118519).

The study tested different parcellation pipelines for tractography-derived putative Vim identification.
Thalamic parcellation was performed on a high quality, multi-shell dataset and a downsampled, clinical-like dataset using two different diffusion signal modeling techniques and two different voxel classification criteria.
Of the resulting four parcellation pipelines, the most reliable  in terms of inter-subject variability has been picked and parcels putatively corresponding to motor thalamic nuclei have been selected by calculating similarity with a histology-based mask of Vim. The effect of data quality and parcellation pipelines on a volumetric index of connectivity clusters has been assessed.

For the different non-invasive parcellations, the study investigates
    1. reliability across subjects
    2. anatomical plausibility wrt the histological mask from Morel/Krauth, NeuroImage 2010
    3. relationship to clinically relevant tremor stimulation targets

Main findings include:
    1. CSD+THR pipeline is most reliable
    2. Precentral SMA and dentatus coincide most with the histological masks
    3. dentatus-linked thalamus voxels are closest to optimal stimulation points from the literature; however with high variability
    4. volumina of parcels vary significantly per pipeline and data quality, thus target planning should not be done purely from the tractography-parcellation results
"""

# %%
import siibra
from nilearn import plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint
from shapely import concave_hull, geometry, intersection
import re
import json
import requests
import nibabel as nib
from nilearn.image import resample_to_img

# %%
# 1. Identify cluster maps coinciding with VIM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The study computes DICE scores of the extracted thalamic clusters to
# histology-based atlas masks from the DISTAL atlas, to evaluate overlap with
# known region delineations. We use the Julich-Brain atlas, wnich provides
# probabilistic maps of these subregions, to re-assess these overlap scores.

# %%
# ---------------------
# 1.1 Load cluster maps
# ---------------------
cluster_names = [
    "Precentral",
    "Dentate",
    "SMA",
    "Paracentral",
    "Postcentral",
    "Temporal",
    "Occipital",
    "Parietal",
    "Prefrontal",
]
hemispheres = {"left": "L", "right": "R"}


# %%
# utility function to load a mask that is resampled to a reference voxel volume
def load_resampled_mask(uri, ref):
    try:
        nim = nib.load(uri)
    except FileNotFoundError:
        response = requests.get(uri)
        with open("temp.nii.gz", "wb") as f:
            f.write(response.content)
        nim = nib.load("temp.nii.gz")
    return resample_to_img(nim, ref, interpolation="nearest")


# %%
# retrieve the cluster maps from the original resource
url_scheme = (
    "https://github.com/BrainMappingLab/"
    "Thalamic-Structural-Connectivity-Based-Parcellation-for-non-invasive-Vim-Identification/"
    "raw/refs/heads/main/{}h/{}_{}_CSDTHR_HQ.nii.gz"
)

clustermaps = {}
mni_template = siibra.get_template("mni152").fetch()
for clustername in cluster_names:
    for hemname, hemcode in hemispheres.items():
        url = url_scheme.format(hemcode.lower(), hemcode, clustername)
        nimg = load_resampled_mask(url, mni_template)
        clustermaps[clustername, hemcode] = siibra.volumes.from_nifti(
            nimg, space="mni152", name=f"{clustername} {hemname}"
        )

# %%
# Hemispheres fo Dentate cluster maps are confused in the study data
tmp = clustermaps["Dentate", "L"]
clustermaps["Dentate", "L"] = clustermaps["Dentate", "R"]
clustermaps["Dentate", "R"] = tmp


# %%
# -------------------------------------------
# 1.2 Reproduce DICE scores with DISTAL atlas
# -------------------------------------------
# The paper reports on average DICE scores with each subject in the individual
# subject spaces.  We fill the reported values in a dataframe as a starting
# point.

# %%
# reported average within-subject scores in the paper
cluster_scores = pd.DataFrame(
    [
        ["Precentral", "L", 0.25],
        ["Precentral", "R", 0.30],
        ["Dentate", "L", 0.27],
        ["Dentate", "R", 0.27],
        ["SMA", "L", 0.22],
        ["SMA", "R", 0.20],
        ["Paracentral", "L", 0.18],
        ["Paracentral", "R", 0.20],
        ["Postcentral", "L", 0.08],
        ["Postcentral", "R", 0.11],
        ["Prefrontal", "L", 0.14],
        ["Prefrontal", "R", 0.12],
        ["Parietal", "L", 0.01],
        ["Parietal", "R", 0.00],
        ["Temporal", "L", 0.00],
        ["Temporal", "R", 0.00],
        ["Occipital", "L", 0.00],
        ["Occipital", "R", 0.00],
    ],
    columns=["Cluster name", "Hemisphere", "Dice score (study)"],
)
cluster_scores.set_index(["Cluster name", "Hemisphere"], inplace=True)
cluster_scores

# %%
# -----------------------------------------------------------------------
# 1.3 Assign clustermaps to Julich-Brain via probability maps correlation
# -----------------------------------------------------------------------

# %%
# Perform probabilistic assignment for each clustermap
julichbrain = siibra.parcellations.get("julich 3.1")
pmaps = julichbrain.get_map(space="mni152", maptype="statistical")
assignments = {}
for (clustername, hemcode), clustermap in clustermaps.items():
    with siibra.QUIET:
        df = pmaps.assign(clustermap, split_components=False)
    df.set_index("region", inplace=True)
    df.sort_values(by="correlation", ascending=False, inplace=True)
    assignments[clustername, hemcode] = df


# %%
# Extract vim correlations and maximum correlations from the
# probabilistic assignments and update  cluster scores.
vim_regions = {c: julichbrain.get_region(f"vim {n}") for n, c in hemispheres.items()}
vim_correlations = {}  # correlation of each clustermap with Julich-Brain VIM
best_correlations = (
    {}
)  # best correlation of each clustermap with any Julich-Brain regiob
best_regions = (
    {}
)  # Name of Julich-Brain region with best correlation to each clustermap
for (clustername, hemcode), df in assignments.items():
    # collect highest score and corresponding region
    best_regions[clustername, hemcode] = df.iloc[0].name
    best_correlations[clustername, hemcode] = df.iloc[0].loc["correlation"]
    vim = vim_regions[hemcode]
    if vim in df.index:
        vim_correlations[clustername, hemcode] = df.loc[vim].correlation.iloc[0]
    else:
        vim_correlations[clustername, hemcode] = 0

# add results to assignment table
cluster_scores["Julich-Brain Correlation (VIM)"] = vim_correlations
cluster_scores["Julich-Brain Correlation (best)"] = best_correlations
cluster_scores["Julich-Brain region"] = best_regions
cluster_scores.round(2)

# %%
# Observations
# ------------
# 1. By filtering strong correlations to the VIM probability map in Julich-Brain,
# we can confirm the cluster selection from the paper. Precentral and Dentate
# are even more distinct than the DICE scores in the paper.
# 2. For Precentral, our assignment suggests that the correlation is stronger to
# the neighboring nucleus VLP rather than the VIM.

# %%
# Bar plot of Distal dice scores versus Julich-Brain correlations
cluster_scores.plot(
    kind="bar",
    y=["Julich-Brain Correlation (VIM)", "Dice score (study)"],
    grid=True,
    figsize=(7, 3),
    width=0.8,
    title="Cluster assignment to VIM",
)
plt.legend(
    [
        "Correlation with Julich-Brain probabilistic maps",
        "Dice score with Distal atlas (study)",
    ],
    loc="center left",
    bbox_to_anchor=(0.15, -1.0),
)
plt.xlabel(None)

# %%
# -----------------------------------------------
# 1.4 Investigate the situation in histology data
# -----------------------------------------------
# We use siibra to retrieve 1-micrometer histology sections in the regions of
# interest, and to compare the fit of map contours with the actual histology.
# We show projected contours of
# 1. the clustermap,
# 2. the DISTAL VIM map,
# 3. Julich-Brain probability maps of VIM, and
# 4. Julich-Brain probability map of the best assigned region.

# %%
# load reference contour in section 3797
with open("e2ec8c09.sands.json", "r") as f:
    d = json.load(f)
vim_reference = siibra.PointCloud(
    [[v["value"] for v in c] for c in d["coordinates"]], space="bigbrain"
)


# %%
def nonzero_points(volume, thres=0):
    # return siibra pointcloud of nonzero points in the volume
    img = volume.fetch()
    arr = img.get_fdata()
    coords = np.argwhere(arr > thres)
    if len(coords) == 0:
        return None
    return siibra.PointCloud(coords, labels=arr[tuple(coords.T)]).transform(
        img.affine, space=volume.space
    )


def coronal_contour(pointcloud, y, ratio=0.7):
    if pointcloud is None:
        return None
    # get the 2D contour of the pointcloud in the given y plane
    hull = concave_hull(
        MultiPoint([[x_, z_] for x_, y_, z_ in pointcloud if abs(y_ - y) <= 1.0]),
        ratio=ratio,
    )
    if len(hull.exterior.coords) == 0:
        return None
    return siibra.PointCloud(
        [(x, y, z) for x, z in hull.exterior.coords], space=pointcloud.space
    )


def get_best_section(volume, thres=0):
    pts_bb = nonzero_points(volume, thres).warp("bigbrain")
    sections = siibra.features.get(
        pts_bb.boundingbox, siibra.features.cellular.CellbodyStainedSection
    )
    point_intersections = [
        pts_bb.intersection(s.get_boundingbox().zoom(2)) for s in sections
    ]
    best_index = np.argmax(
        [0 if pts is None else len(pts) for pts in point_intersections]
    )
    return sections[best_index]


# %%
# get a clustermap
clustername = "Precentral"
hem = "left"
hemcode = hemispheres[hem]
clustermap = clustermaps[clustername, hemcode]

plt.figure()
plotting.plot_glass_brain(clustermap.fetch())
# plt.title(f"{clustername} {hem}")

df = assignments[clustername, hemcode].query("correlation > 0.1")[["correlation"]]
df.index = [re.sub(r"\s*\(.*?\)", "", r.name) for r in df.index]
df.plot(kind="bar", figsize=(2.5, 2), grid=True, title=f"{clustername} {hem}")

# %%
# specify a bigbrain section for the figure
best_section = get_best_section(clustermap)
ref_section = [
    s
    for s in siibra.features.cellular.CellbodyStainedSection._get_instances()
    if "3797" in s.name
][0]
section = ref_section
print(f"Section: {section.name[1:5]}")


# %%
shortname = lambda n: re.sub(r"\s*\(.*\)", "", n)
bigbrain_contour = lambda m, y, t: coronal_contour(
    nonzero_points(m, t).warp("bigbrain"), y, ratio=0.5
)

# %%
# define and extract patch
patch_res_mm = 0.02
y_bigbrain = section.get_boundingbox().center[1]

# %%
# Uncomment this code if you have a license for DISTAL Atlas (Ewert 2017)
# `distalmap = nib.load("pathtodistalmap")`
# `vim_map_distal = distalmap[hemcode]`

# %%
# map contours
vim_map_jba = julichbrain.get_region(f"vim {hem}").get_regional_map("mni152")
best_region_jba = cluster_scores.loc[clustername, hemcode]["Julich-Brain region"]
best_map_jba = best_region_jba.get_regional_map("mni152")
contours = {
    f"Cluster {clustername} {hem}": (
        "k",
        bigbrain_contour(clustermap, y_bigbrain, 0.0),
    ),
    # f"VIM {hem} (Distal)": ("m", bigbrain_contour(vim_map_distal, y_bigbrain, 0.5)),  # Uncomment this code if you have a license for DISTAL Atlas (Ewert 2017)
    f"{shortname(best_region_jba.name)} (Julich-Brain)": (
        "b",
        bigbrain_contour(best_map_jba, y_bigbrain, 0.5),
    ),
    f"VIM {hem} (Julich-Brain)": ("r", bigbrain_contour(vim_map_jba, y_bigbrain, 0.5)),
}
if hem == "left" and "3797" in section.name:
    contours["VIM reference annotation"] = ("g", vim_reference)

# %%
# patch extraction
x0, _, z0 = np.array([c.coordinates.min(0) for _, c in contours.values() if c is not None]).min(0) - 1
x1, _, z1 = np.array([c.coordinates.max(0) for _, c in contours.values() if c is not None]).max(0) + 1
y0 = section.get_boundingbox().minpoint[1]
y1 = section.get_boundingbox().maxpoint[1]
voi = siibra.BoundingBox([x0, y0, z0], [x1, y1, z1], space="bigbrain")
patch = section.fetch(voi=voi, resolution_mm=0.02)

# %%
# whole brain view of BigBrain section
tpl = siibra.get_template(voi.space).fetch(resolution_mm=0.8)
plt.figure()
view = plotting.plot_img(
    patch, bg_img=tpl, cmap="gray", title=f"Section {section.name[1:5]}"
)
for color, cont in contours.values():
    if cont is not None:
        view.add_markers(cont.as_list(), marker_size=0.2, marker_color=color)

# %%
# detailed patch view
plt.figure()
plt.imshow(patch.get_fdata().squeeze(), cmap="gray")
legend = []
for name, (color, cont) in contours.items():
    if cont is not None:
        Vx, Vy, Vz = np.dot(np.linalg.inv(patch.affine), cont.homogeneous.T)[:3]
        plt.plot(Vz, Vx, color=color)
        legend.append(name)
# plt.axis('off')
plt.legend(legend, loc="center left", bbox_to_anchor=(1, 0.5))

# %%
x0, _, z0 = vim_reference.coordinates.min(0) - 1
x1, _, z1 = vim_reference.coordinates.max(0) + 1
ref_voi = siibra.BoundingBox([x0, y0, z0], [x1, y1, z1], space="bigbrain")
ref_patch = section.fetch(voi=ref_voi, resolution_mm=-1)

# %%
plt.figure()
plt.imshow(ref_patch.get_fdata().squeeze(), cmap="gray")
Vx, Vy, Vz = np.dot(np.linalg.inv(ref_patch.affine), vim_reference.homogeneous.T)[:3]
plt.plot(Vz, Vx, color="g", lw=3)

y_, x_ = 1150, 3400
w = 1000
plt.plot([x_, x_, x_ + w, x_ + w, x_], [y_, y_ + w, y_ + w, y_, y_], "k-")

# %%
plt.figure()
plt.imshow(ref_patch.get_fdata().squeeze()[y_:y_ + w, x_:x_ + w], cmap="gray")
Vx, Vy, Vz = np.dot(np.linalg.inv(ref_patch.affine), vim_reference.homogeneous.T)[:3]
polygon = geometry.Polygon(np.array([Vz - x_, Vx - y_]).T)
canvas = geometry.Polygon([[0, 0], [0, w], [w, w], [w, 0], [0, 0]])
X, Y = intersection(polygon, canvas).exterior.xy
plt.plot(X, Y, color="g", lw=6)
plt.axis("off")
