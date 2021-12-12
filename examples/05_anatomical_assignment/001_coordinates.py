"""
Assigning coordinates to brain regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`siibra` can use continuous parcellations maps to make a probabilistic assignment of exact and imprecise coordinates to brain regions.
We start by selecting the Julich-Brain probabilistic maps from the human atlas, which we will use for the assignment. 
"""
import siibra
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
with siibra.QUIET:
    julich_pmaps = atlas.get_map(
        space="mni152",
        parcellation="julich",
        maptype="continuous"
    )

# %%
# **Assigning exact coordinate specifications.**
# We now determine the probability values of cytoarchitectonic regions at a
# an arbitrary point in MNI reference space. The example point is manually chosen. 
# It should be located in PostCG of the right hemisphere.
# For more information about the ``siibra.Point`` class see :ref:`locations`.
# The output is a pandas dataframe, which includes the values of different probability
# maps at the given location.
# We can sort the table by these values, to see that the region with highest probability is
# indeed the expected region. 
point = siibra.Point((27.75, -32.0, 63.725), space='mni152')
assignments = julich_pmaps.assign(point)
assignments.sort_values(by=['MaxValue'], ascending=False)

# %%
# **Assigning coordinate specifications with location uncertainty.**
# Typically, coordinate specifications are not exact. For example, we obtain a 
# position from an sEEG electrode, which has several millimeters of uncertainty
# (especially after warping it to standard space!). If we specify the position
# with a location uncertainty, ``siibra`` will not just read out the values of 
# the probability maps, but instead generate a 3D Gaussian blob with a corresponding
# standard deviation, and correlate the 3D blob with the maps. We then obtain
# the average of the map values over the blob, but also additional measures of
# comparison: A correlation coefficient, the intersection over union (IoU), 
# a containedness score of the blob wrt. the region, and a containedness score
# of the region wrt. the blob. Per default, the resulting table is sorted
# by correlation coefficient. Here, we filter to show only the rows with 
# a containedness score of at least 0.5
point_uncertain = siibra.Point((27.75, -32.0, 63.725), space='mni152', sigma_mm=5.)
assignments = julich_pmaps.assign(point_uncertain)
assignments[assignments.Contained>=0.5]

# %%
# To verify the result, we plot the assigned probability maps at the requested position.
from nilearn import plotting
for index, assignment in assignments[assignments.Contained>=0.5].iterrows():
    pmap = julich_pmaps.fetch(mapindex=assignment.MapIndex)
    plotting.plot_stat_map(pmap, cut_coords=tuple(point), title=assignment.Region)
