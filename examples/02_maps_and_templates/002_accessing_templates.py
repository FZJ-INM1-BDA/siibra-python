"""
.. _templates:

Access brain reference templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
# Here we use the plotting tools provided by `nilearn <https://nilearn.github.io>`_
from nilearn import plotting
plotting.view_img(img, bg_img=None, cmap='gray')

