"""
Assigning brain regions to coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`siibra` can use continuous parcellations maps to make a probabilistic assignment of exact and imprecise coordinates to brain regions.
"""


# %%
# We start by selecting the Julich-Brain probabilistic maps from the human
# atlas, which we will use for the assignment. 
import siibra
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
julich_pmaps = atlas.get_map(
    space="mni152",
    parcellation="julich",
    maptype="continuous"
)

# %%
# We now determine the probability values of cytoarchitectonic regions at a
# an arbitrary point in MNI reference space. The example point is manually chosen. 
# It should be located in PostCG of the right hemisphere.
# For more information about the ``siibra.Point`` class see :ref:`locations`.
point = siibra.Point((27.75, -32.0, 63.725), space='mni152')
assignments = julich_pmaps.assign(point)
assignments

