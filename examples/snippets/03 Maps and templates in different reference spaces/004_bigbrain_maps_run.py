"""
High-resolution maps
~~~~~~~~~~~~~~~~~~~~

Similar as for the BigBrain reference template (cf. :ref:`templates`),
siibra provides access to high-resolution parcellation maps defined in the 
20 micrometer BigBrain space. These maps can be retrieved at reduced
resolution for the whole brain, and  at full resolution by specifying regions
of interest.
"""

# %%
# We start by selecting an atlas.
import siibra
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS

# %%
# To access BigBrain maps at high resolution, we specify a rectangular volume
# of interest spanned by two 3D points in physical coordinates.
voi = siibra.BoundingBox(
    "-30.590mm, 3.270mm, 47.814mm",
    "-26.557mm, 6.277mm, 50.631mm",
    space=atlas.spaces.BIG_BRAIN
)

# %%
# We first extract the corresponding chunk from the BigBrain template.
bigbrain_chunk = atlas.get_template('bigbrain').fetch(resolution_mm=0.02, voi=voi)
bigbrain_chunk

# %%
# Next we select a parcellation which provides a map for BigBrain, and extract
# labels for the same volume of interest. We choose the cortical layer maps by Wagstyl et al. 
# Note that by specifyin "-1" as a resolution, `siibra` will fetch the highest
# possible resolution.
layermap = atlas.get_map(space='bigbrain', parcellation='layers')
mask = layermap.fetch(resolution_mm=-1, voi=voi)
mask

# %%
# Since we operate in physical coordinates, we can plot both image chunks
# superimposed, even if their resolution is not exactly identical.
from nilearn import plotting
plotting.view_img(mask, bg_img=bigbrain_chunk, opacity=.1, symmetric_cmap=False)

# %%
# `siibra` can help us to assign a brain region to the position of the volume
# of interest. This is covered in more detail in :ref:`assignment`. For now,
# just note that `siibra` can employ spatial objects from different template spaces. 
# Here it automatically warps the centroid of the volume of interst to MNI space
# for location assignment. 
julich_pmaps = atlas.get_map(space='mni152', parcellation='julich', maptype='continuous')
assignments = julich_pmaps.assign_coordinates(voi.center)
for region, mapindex, scores in assignments:
    print(f"{region.name:40.40} {scores['correlation']:4.2f}")

