"""
Accessing BigBrain image data and maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

siibra provides access to high-resolution image data parcellation maps defined
for the 20 micrometer BigBrain space. 
The BigBrain is very different from other templates. Its native resolution is
20 micrometer, resulting in about one Terybyte of image data. Yet, fetchig the
template works the same way as for the MNI templates, with the difference that
we can specify a reduced resolution or volume of interest to fetch a feasible
amount of image data, or a volume of interest.
"""

# %%
# We start by selecting an atlas.
import siibra
from nilearn import plotting
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS

# %%
# Per default, `siibra` will fetch the whole brain volume at a reasonably
# reduced resolution.
bigbrain = atlas.get_template('bigbrain')
bigbrain_whole = bigbrain.fetch()
plotting.view_img(bigbrain_whole, bg_img=None, cmap='gray')

# %%
# To see the full resolution, we may specify a bounding box in the physical
# space. You will learn more about spatial primities like points and bounding
# boxes in :ref:`locations`. For now, we just define a volume of interest from
# two corner points in the histological space. We specify the points with 
# a string representation, which could be conveniently copy pasted from the
# interactive viewer "siibra explorer", hosted at https://atlases.ebrains.eu/viewer.
# Of course we can also specify coordinates by a 3-tuple, and in other ways.
voi = siibra.BoundingBox(
    point1="-30.590mm, 3.270mm, 47.814mm",
    point2="-26.557mm, 6.277mm, 50.631mm", 
    space=siibra.spaces['bigbrain']
)
bigbrain_chunk = bigbrain.fetch(voi=voi, resolution_mm=0.02)
plotting.view_img(bigbrain_chunk, bg_img=None, cmap='gray')

# %%
# Note that since both fetched image volumes are spatial images with a properly
# defined transformation between their voxel and physical spaces, we can
# directly plot them correctly superimposed on each other:
plotting.view_img(bigbrain_chunk, bg_img=bigbrain_whole, cmap='magma', cut_coords=tuple(voi.center))

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
for region, mapindex, probability in assignments:
    print(f"{region.name:40.40} {probability:4.2f}")

