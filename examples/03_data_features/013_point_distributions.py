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
.. _013_point_distributions
:bdg-secondary:`Intermediate`

Cell and tracer point distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some datasets provide the positions of individual cells or tracer labels as
coordinates in a reference space. `siibra` provides them as point
distributions, whose data is a table with one row per point. The columns x, y
and z hold RAS+ millimetres in the reference space; stored coordinates are
mapped to them when the data is read.
"""

# %%
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nilearn import plotting

import siibra


# %%
# To summarize several point distributions, we compute a kernel density
# estimate (KDE) of each one on a common grid covering the template of their
# space, and average the KDEs into one. Each KDE sums to one, so every point
# distribution contributes equally, whatever its number of points.
def mean_kde(point_distributions, voxel_mm):
    space = point_distributions[0].space
    bbox = space.get_template().get_boundingbox(clip=False)
    minpoint = np.array(bbox.minpoint.coordinate)
    shape = np.ceil((np.array(bbox.maxpoint.coordinate) - minpoint) / voxel_mm).astype(int)
    affine = np.diag([voxel_mm, voxel_mm, voxel_mm, 1.0])
    affine[:3, 3] = minpoint
    grid = siibra.volumes.from_array(np.zeros(shape, dtype="float32"), affine, space=space.name)
    kdes = []
    for point_distribution in point_distributions:
        points = siibra.locations.PointCloud(
            point_distribution.data[["x", "y", "z"]].to_numpy(), space=space, sigma_mm=voxel_mm
        )
        kde = siibra.volumes.from_pointcloud(points, target=grid, clip_out_of_bounds=True)
        kdes.append(kde.fetch().get_fdata())
    return nib.Nifti1Image(np.mean(kdes, axis=0), affine)


def plot_on_template(kde, space, title):
    plotting.plot_stat_map(
        kde,
        bg_img=space.get_template().fetch(),
        cut_coords=plotting.find_xyz_cut_coords(kde),
        cmap="magma",
        title=title,
        draw_cross=False,
    )


# %%
# Every cell distribution belongs to one subject. Datasets that are stored per
# region give one cell distribution per subject and region. Here, we look at
# the accessory olfactory bulb of the Allen Mouse Common Coordinate Framework.
allen = siibra.parcellations.get("Allen Mouse Common Coordinate Framework v3 2017")
ccf = siibra.spaces.get("Allen Mouse Common Coordinate Framework v3")
aob = allen.get_region("Accessory olfactory bulb")
cell_distributions = siibra.features.get(aob, siibra.features.cellular.CellDistribution)
for cell_distribution in cell_distributions:
    print(cell_distribution.name, "|", cell_distribution.modality)

# %%
# We select the cell distribution of subject 81264. Its data has one row per
# cell.
cell_distribution_81264 = next(
    cell_distribution
    for cell_distribution in cell_distributions
    if cell_distribution.subject == "81264"
)
cell_distribution_81264.data.head()

# %%
# The cells can be shown in 3D inside the surface mesh of the template.
mesh = ccf.get_template().fetch(format="mesh")
cells = cell_distribution_81264.data
figure = go.Figure([
    go.Mesh3d(
        x=mesh["verts"][:, 0], y=mesh["verts"][:, 1], z=mesh["verts"][:, 2],
        i=mesh["faces"][:, 0], j=mesh["faces"][:, 1], k=mesh["faces"][:, 2],
        color="lightgray", opacity=0.15, name=ccf.name,
    ),
    go.Scatter3d(
        x=cells["x"], y=cells["y"], z=cells["z"],
        mode="markers", marker={"size": 2, "color": "crimson"},
        name=f"subject {cell_distribution_81264.subject}",
    ),
])
figure.update_layout(scene_aspectmode="data", title=cell_distribution_81264.name)
figure.show()

# %%
# The averaged KDE of all subjects' cells, on the Allen mouse brain template.
plot_on_template(
    mean_kde(cell_distributions, voxel_mm=0.1),
    ccf,
    title=f"Cells in {aob.name}, mean of {len(cell_distributions)} subjects",
)

# %%
# Point distributions are also anchored to their reference space, so a space
# query finds all of them in that space, including whole-brain tables that
# have no region. Reading them all would download every file, so we only
# count them per subject here.
cell_distributions_in_ccf = siibra.features.get(ccf, siibra.features.cellular.CellDistribution)
overview = pd.DataFrame(
    {"subject": [cell_distribution.subject for cell_distribution in cell_distributions_in_ccf]}
)
overview.groupby("subject").size()

# %%
# Tracer distributions are features of single experiments. They are anchored
# to their injection site, so a region query finds experiments injected
# there, while their labels may lie anywhere. Here, injections into the
# secondary motor area of the Waxholm Space rat brain atlas. The injection
# site lists the regions it is specified as, and its location if known.
waxholm = siibra.parcellations.get("waxholm v4")
m2 = waxholm.get_region("Secondary motor area")
tracer_distributions = siibra.features.get(m2, siibra.features.connectivity.TracerDistribution)
for tracer_distribution in tracer_distributions:
    site = tracer_distribution.injection_site
    print(tracer_distribution.name, "|", tracer_distribution.modality, "|", [r.name for r in site.regions])

# %%
# The averaged KDE of the labels of these experiments, on the Waxholm rat
# brain template. It shows where their labels lie; whether these are
# projections from or to the injection site depends on the tracers, so
# experiments are best averaged per tracer, as named by their modality.
waxholm_space = tracer_distributions[0].space
plot_on_template(
    mean_kde(tracer_distributions, voxel_mm=0.2),
    waxholm_space,
    title=f"Labels after injections into {m2.name}, mean of {len(tracer_distributions)} experiments",
)
