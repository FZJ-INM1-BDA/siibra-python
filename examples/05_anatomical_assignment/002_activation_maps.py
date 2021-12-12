"""
Assign modes in activation maps to brain regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Continuous parcellations maps can also assign regions to image volumes. 
If given a ``Nifti1Image`` object as input, the assignment method will interpret it as a measurement of a spatial distribution.
It will first split the image volume into disconnected components, i.e. any subvolumes which are clearly separated by zeros.
Then, each component will be compared to each continuous maps in the same way that the Gaussian blobs representing uncertain points are processed in 
:ref:`sphx_glr_examples_05_anatomical_assignment_001_coordinates.py`.

We start again by selecting the Julich-Brain probabilistic maps from the human atlas, which we will use for the assignment. 
"""
import siibra
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
with siibra.QUIET: # suppress progress output here
    julich_pmaps = atlas.get_map(
        space="mni152",
        parcellation="julich",
        maptype="continuous"
    )

# %%
# As an exemplary input signal, we threshold a continuous map from the
# functional maps (DiFuMo 64) by Thirion et al.
with siibra.QUIET: # suppress progress output here
    difumo_maps = atlas.get_map(
        space='mni152',
        parcellation='difumo 64', 
        maptype='continuous'
    )
from nilearn import image
img = image.threshold_img(difumo_maps.fetch(15), 0.0004)
region = difumo_maps.decode_label(15)

# let's look at the resulting image
from nilearn import plotting
plotting.view_img(img, 
    title="Thresholded functional map of {region.name}",
    colorbar=False
)

# %%
# We now assign Julich-Brain regions to this functional map with multiple
# modes. Note that when given an image as input, the ``assign`` method
# will output an image volume with labels for the detecteded components,
# so we can lookup which component corresponds to which part of the 
# input volume.
# Since we are here ususally interested in correlations of the modes,
# we filter the result by significant (positive) correlations.
with siibra.QUIET: # suppress progress output
    assignments, components = julich_pmaps.assign(img)
assignments[assignments.Correlation>=0.2]


