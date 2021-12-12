"""
Find predefined reference spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Just as atlas and parcellation objects, `siibra` provides a registry of predefined
reference space objects that will be bootstrapped when you load the library for
the first time, and stay in your local file cache for future use. Reference
spaces are semantic objects which define a brain coordinate system. Associated
to each reference space are one or more reference templates, representing a
concrete reference image or surface representation of the brain.
"""

# %%
# As for atlases and parcellations, siibra provides a registry of predefined
# reference spaces:
import siibra
dir(siibra.spaces)

# %%
# Fetching an object works in the same way as for e.g. `siibra.atlases` (see :ref:`atlases`)
siibra.spaces['mni152']

# %%
# Typically however, we are only interested in the reference spaces supported by
# a given atlas. Atlases provide their own space registry for this
# purpose, which includes the relevant subset of the spaces.
atlas = siibra.atlases['human']
dir(atlas.spaces)

# %%
# These can be used like any other registry object:
atlas.spaces['colin27']

# %%
# We can also explicitely request a supported space object from the
# atlas, which has the same effect as accessing the Registry.
atlas.get_space('bigbrain')

