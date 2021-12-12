"""
Colorizing a map with neurotransmitter receptor densities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parcellation maps can be colorized with a dictionary of values.
This is useful to visualize data features in 3D space.
Here we demonstrate it for average densities of the GABAA receptor.
"""

# %%
# We start by selecting an atlas and the Julich-Brain parcellation.
import siibra
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
jubrain = atlas.get_parcellation('julich 2.9')

# %%
# As in the previous example, we extract all receptor density features linked
# to Julich-Brain.
receptor_features = siibra.get_features(jubrain, siibra.modalities.ReceptorDistribution)

# %%
# Colorizing a map requires a dictionary that maps region objects to numbers.
# We build a dictionary mapping Julich-Brain regions to average receptor
# densities measured for GABAA.
receptor = 'GABAA'
mapping = {
    jubrain.decode_region(f.regionspec) : f.fingerprint[receptor].mean
    for f in receptor_features if receptor in f.fingerprint.labels
}

# %%
# Now colorize the Julich-Brain maximum probability map and plot it.
colorized_map = jubrain.get_map(space='mni152').colorize(mapping)
from nilearn import plotting
plotting.plot_stat_map(
    colorized_map, cmap='viridis',
    title=f"Average densities available for {receptor}"
)

