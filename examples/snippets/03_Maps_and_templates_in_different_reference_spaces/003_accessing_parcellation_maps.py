"""
Accessing parcellation maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Just like reference templates (covered in :ref:`templates`), a parcellation map
or region map is a spatial object - usually a volumetric image with discrete
voxel labels for each region, a binary mask image, or a 3D distribution of
float values in case of a probabilistic map. Since some parcellations provide maps
in different reference spaces, obtaining a parcellation map involves to specify
a reference space.
"""

# %%
# We start by selecting an atlas.
import siibra
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS

# %%
# We select the maximum probability map of Julich-Brain in MNI152 space,
# which is a parcellation map with discrete labels. The `get_map` method
# of the atlas assumes maptype='labelled' by default.
julich_mpm = atlas.get_map(space="mni152", parcellation="julich", maptype="labelled")
julich_mpm

# %%
# As already seen for reference templates, the returned object does not contain
# any image data yet, since siibra uses a lazy strategy for loading data. 
# To access the actual image data, we call the fetch() method. 
# This gives us a Nifti1Image object defined in the reference space.
from nilearn import plotting
mapimg = julich_mpm.fetch()
print(type(mapimg))
plotting.plot_roi(mapimg, title=f"{julich_mpm.parcellation.name}")

# %%
# This map shows only the left hemisphere, because Julich-Brain ships the left
# and right hemispheres in different volumes, so corresponding regions can use
# the same label index while still being distinguishable. We can find the number
# of mapped volumes inside a parcellation map simply from its length:
len(julich_mpm)

# %%
# To get the second labelled volume, we can explicitly pass the map index to
# `fetch()` - it defaults to 0 otherwise.
mapimg2 = julich_mpm.fetch(mapindex=1)
plotting.plot_roi(mapimg2, title=f"{julich_mpm.parcellation.name}, map index 1")

# %%
# Instead of dealing with map indices, we can also simply iterate over all
# available maps using `fetch_iter()`.
for img in julich_mpm.fetch_iter():
    plotting.plot_stat_map(img)

# %%
# Julich-Brain, like some other parcellations, provides probabilistic maps.
# The maximum probability map is just a simplified representation, displaying
# for each voxel the label of the brain region with highest probability. We can
# access the probabilistic information by requesting the
# "continuous" maptype (`siibra.maptype.CONTINUOUS`).
# Note that since the set of probability maps are usually a large number of
# sparsely populated image volumes, `siibra` will load the volumetric data only
# once and then convert it to a sparse index format, that is much more
# efficient to process and store. The sparse index is cached on the local disk,
# therefore subsequent use of probability maps will be much faster.
julich_pmaps = atlas.get_map(space="mni152", parcellation="julich", maptype="continuous")
julich_pmaps

# %%
# Since the continuous maps overlap, this map provides access to several
# hundreds of brain volumes.
print(len(julich_pmaps))

# %%
# Again, we can iterate over all >300 pmaps using `fetch_iter_()`, but for simplicity, we just access a random index here.
pmap = julich_pmaps.fetch(mapindex=102)
plotting.plot_stat_map(pmap)

# %%
# Now we might wonder which region this map index actually refers to. 
# The Parcellation map objects can safely translate
# indices into regions:
julich_pmaps.decode_label(mapindex=102)

# %%
# Vice versa, we can find the parcellation index of a given region:
julich_pmaps.decode_region('hoc5 left')

# %%
# Besides parcellation maps, `siibra` can also produce binary masks of brain regions.
v5l = atlas.get_region('hoc5 left')
v5l_mask = v5l.build_mask("mni152")
plotting.plot_roi(v5l_mask)


