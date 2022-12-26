# Copyright 2018-2021
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
.. _templates:

Accessing brain reference templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Other than the reference space itself, a reference template is a spatial object -
more precisely an image volume or surface mesh. Since reference templates are directly
linked to their corresponding brain reference space, we access templates by specifying
the space. Like all image and mesh objects in `siibra`, reference template implement lazy
loading: The actual image or mesh data is only fetched from the corresponding online
resource when explicitly calling a "fetch" method.
"""


# %%
# We choose the MNI152 space, and then request the template that `siibra`
# linked to it. In this concrete case, it turns out to be a NIfTI file stored
# on a remote server
import siibra
mni152tpl = siibra.spaces.get('mni152').get_template()
type(mni152tpl)

# %%
# Alternatively, a template can be retrieved from an atlas,
# or right away from the siibra package:
mni152tpl = siibra.atlases.get('human').get_template(space="mni152")
mni152tpl = siibra.get_template("icbm 152 asym")

# %%
# To load the actual image object, we use the template's fetch() method.
# This gives us a Nifti1Image object with spatial metadata, compatible with
# the wonderful nibabel library and other common neuroimaging toolboxes.
img = mni152tpl.fetch()
img

# %%
# We can easily display this template with common visualization tools.
# Here we use the plotting tools provided by `nilearn <https://nilearn.github.io>`_
from nilearn import plotting
plotting.view_img(img, bg_img=None, cmap='gray')

# %%
