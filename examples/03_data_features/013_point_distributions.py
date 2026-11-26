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
import siibra

# %%
# Cell distributions are stored per subject and region. A query returns one
# compound feature per subject, whose elements are the regions matching the
# query. Here, we look at the accessory olfactory bulb of the Allen Mouse
# Common Coordinate Framework.
allen = siibra.parcellations.get("Allen Mouse Common Coordinate Framework v3 2017")
aob = allen.get_region("Accessory olfactory bulb")
subjects = siibra.features.get(aob, siibra.features.cellular.CellDistribution)
for cf in subjects:
    print(cf.name, "|", cf.indices)

# %%
# The data of a subject's compound has the cells of all its regions, with the
# region of each cell.
cells = subjects[0].data
print(cells.groupby("region").size())
cells.head()

# %%
# The elements are the cell distributions of single regions. The affine maps
# the stored coordinates, here voxels of the atlas, to the reference space.
element = subjects[0].get_element(subjects[0].indices[0])
print(element.subject, "|", element.region, "|", element.space.name)
print(element.affine)

# %%
# Point distributions are also anchored to their reference space, so a space
# query finds all of them in that space: here, every subject's cells across
# the brain, including whole-brain tables that have no region. Its data would
# read one file per region, so we only count the regions here.
ccf = siibra.spaces.get("Allen Mouse Common Coordinate Framework v3")
for cf in siibra.features.get(ccf, siibra.features.cellular.CellDistribution):
    print(cf.name, "|", len(cf), "regions")

# %%
# The coordinates can be plotted with pandas, here projected on the x and y
# axes.
cells.plot.scatter(x="x", y="y", s=1, title=subjects[0].name)

# %%
# Tracer distributions are features of single experiments; they are not
# compounded. They are anchored to their injection site, so a region query
# finds experiments injected there, while their labels may lie anywhere. Here,
# injections into the secondary motor area of the Waxholm Space rat brain
# atlas.
waxholm = siibra.parcellations.get("waxholm v4")
m2 = waxholm.get_region("Secondary motor area")
experiments = siibra.features.get(m2, siibra.features.connectivity.TracerDistribution)
for f in experiments:
    print(f.name, "|", f.modality)

# %%
# The injection site lists the regions it is specified as, and its location
# if known.
site = experiments[0].injection_site
print([r.name for r in site.regions], site.location)
experiments[0].data.head()
