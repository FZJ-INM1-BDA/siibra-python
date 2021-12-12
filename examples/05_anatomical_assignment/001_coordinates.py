"""
Assigning brain regions to coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`siibra` can use continuous parcellations maps to make a probabilistic assignment of exact and imprecise coordinates to brain regions.
"""


# %%
# We start by selecting an atlas, and specifying a set of points in a reference
# space. For more information about the ``siibra.PointSet`` class, see :ref:`locations`.
import siibra
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
points = siibra.PointSet(
    [
        (31.0, -89.6, -6.475),  # a real sEEG contact point
        (27.75, -32.0, 63.725)  # a point manually selected in PostCG, right hemisphere
    ], space='mni152'
)

# %%
# To perform the assignment, we use the Julich-brain probabilistic cytoarchitectonic maps. 
julich_pmaps = atlas.get_map(
    space="mni152",
    parcellation="julich",
    maptype="continuous")

# %%
# We assume the point locations to be imprecise, and specify an error tolerance of 5mm.
# The assignment will then use a 3D Gaussian kernel with this bandwidth instead of 
# a fixed point. 
assignments = julich_pmaps.assign_coordinates(points, sigma_mm=5.)
print(assignments)

# %%
# The assignment returns a list of regions which matched each coordinate,
# together with a tuple of scores including correlations and containedness. 
# The latter measures the degree that the given point is located inside a region, 
# and we will use it to filter assigned regions to the ones that like include
# the point.
from nilearn import plotting 
for region, mapindex, scores in assignments[0]:
    if scores['contains']>.5:
        plotting.plot_stat_map(
            julich_pmaps.fetch(mapindex),
            cut_coords=tuple(points[0]),
            title=f"{region.name} ({scores['contains']*100:.1f}%)"
        )

# %% 
# As an application example, we can retrieve structural connectivity profiles for the regions that were assigned to the two points, and use them to study the connection strengh between the regions.

# get profiles for the top assigned region
closest_region, mapindex, scores = next(iter(assignments[0]))
profiles = siibra.get_features(
    closest_region,
    siibra.modalities.ConnectivityProfile
)

# %%
# We create plots of connection strength to the 20 most strongly connected regions, for each of the returned profiles. Note that the profiles come from different connectivity datasets. The `src_info` and `src_name` attributes tell us more about each dataset.
# 
# First, we decode the profiles with the parcellation object. This will convert the column names of the connectivity profile to explicit brain region objects, helping to disambiguiate region names.
with siibra.QUIET:
    decoded_profiles = [
        p.decode(closest_region.parcellation)
        for p in profiles]
p = decoded_profiles[0]
target_regions = [region for strength, region in p[:20]]
target_regions

# %%
# Next we define a plotting function for the decoded profiles, which takes the N most strongly connected regions of the first profile, and plot the connection strengths of all found profiles for those N target regions.
import matplotlib.pyplot as plt
def plot_connectivity_profiles(profiles,target_regions):
    # Let's plot the so obtained regions and their strenghts
    N = len(target_regions)
    xticks = range(N)
    fig  = plt.figure()
    ax1  = fig.add_subplot(211)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(
        [r.name for r in target_regions],
        rotation=45, fontsize=10, ha='right')
    for p in profiles:
        probs = {region:prob for prob,region in p}
        y = [probs[r] if r in probs else 0
             for r in target_regions ]
        ax1.plot(xticks,y,'.-',lw=1)
    ax1.grid(True)
    return fig

# %%
# Now we can create the plot.
fig = plot_connectivity_profiles(decoded_profiles,target_regions)
fig.legend([p.name for p in profiles],
           loc='upper left', bbox_to_anchor=(1.05, 1.0),
           prop={'size': 9})
fig.gca().set_title(f"Connection strengths from area {closest_region.name}")
plt.show()

