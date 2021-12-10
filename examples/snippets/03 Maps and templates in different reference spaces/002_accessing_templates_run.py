"""
.. _templates:

Accessing brain reference templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Other than the rerence space itself, a reference template is a spatial object - more precisely an image volume or surface mesh. Since reference templates are directly linked to their corresponding brain reference space, we access templates by specifying the space. Like all image and mesh
objects in `siibra`, instantiation of a reference template is lazy: The actual image or mesh is only loaded after explicitely calling a "fetch" method.
"""


# %%
# We choose the MNI152 space, and then request the template that `siibra`
# linked to it. In this concrete case, it turns out to be a NIfTI file stored
# on a remote server
import siibra
mni152tpl = siibra.spaces['mni152'].get_template()
type(mni152tpl)

# %%
# A more common way to grab a template however is via an atlas. The same as
# above can be achieved this way:
atlas = siibra.atlases['human']
mni152tpl = atlas.get_template(space="mni152")

# %%
# To load the actual image object, we will use the template's fetch() method.
# This gives us a Nifti1Image object with spatial metadata, compatible with
# the wonderful nibabel library and most common neuroimaging toolboxes.
img = mni152tpl.fetch()
img

# %%
# We can directly display this template now with common visualization tools.
# Here we use the plotting tools provided by nilearn.(https://nilearn.github.io)
from nilearn import plotting
plotting.view_img(img, bg_img=None, cmap='gray')

# %%
# The BigBrain template is a very different dataset. Its native resolution is
# 20 micrometer, resulting in about one Terybyte of image data. Yet, fetchig the
# template works the same way, with the difference that we can specify a 
# reduced resolution or volume of interest to fetch a feasible amount of image data.
# Per default, `siibra` will fetch the whole brain volume at a reasonably
# reduced resolution.
bigbrain = atlas.get_template('bigbrain')
bb_whole = bigbrain.fetch()
plotting.view_img(bb_whole, bg_img=None, cmap='gray')

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
bb_chunk = bigbrain.fetch(voi=voi, resolution_mm=0.02)
plotting.view_img(bb_chunk, bg_img=None, cmap='gray')

# %%
# Note that since both fetched image volumes are spatial images with a properly
# defined transformation between their voxel and physical spaces, we can
# directly plot them correctly superimposed on each other:
plotting.view_img(bb_chunk, bg_img=bb_whole, cmap='magma', cut_coords=tuple(voi.center))

