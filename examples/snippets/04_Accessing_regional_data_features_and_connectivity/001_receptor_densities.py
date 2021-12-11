"""
Neurotransmitter receptor densities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EBRAINS provides transmitter receptor density measurments linked to a selection of cytoarchitectonic brain regions in the human brain (Palomero-Gallagher, Amunts, Zilles et al.). These can be accessed by calling the ``siibra.get_features()`` method with the ``siibra.modalities.ReceptorDistribution`` modality (or the shorthand 'receptor'), and by specifying a cytoarchitectonic region. Receptor densities come as a structured datatype which includes a regional fingerprint with average densities for different transmitters, as well as often an additional cortical density profile and a sample autoradiograph patch. They bring their own `plot()` method to produce a quick illustration.
"""


# %%
# We start by selecting an atlas.
import siibra
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS

# %%
# If we query this modality for the whole atlas instead of a particular
# brain region, all linked receptor density features
# will be returned.
features = siibra.get_features( atlas, siibra.modalities.ReceptorDistribution)
print("Receptor density features found for the following regions:")
print("\n".join(f.regionspec for f in features))

# %%
# When providing a particular region instead, the returned list is filtered accordingly. 
# So we can directly retrieve densities for the primary visual cortex:
v1_features = siibra.get_features(atlas.get_region('v1'), 'receptor')
for f in v1_features:
    fig = f.plot()

# %%
# Each feature includes a data structure for the fingerprint, with mean and
# standard values for different receptors.
fp = v1_features[0].fingerprint
for label, mean, std in zip(fp.labels, fp.meanvals, fp.stdvals):
    print(f"{label:20.20} {mean:10.0f} {fp.unit}      +/-{std:4.0f}")

# %%
# Many of the receptor features also provide a profile of density measurements
# at different cortical depths, resolving the change of
# distribution from the white matter towards the pial surface.
# The profile is stored as a dictionary of density measures from 0 to 100%
# cortical depth.
p_ampa = v1_features[0].profiles['AMPA']
import matplotlib.pyplot as plt
plt.plot(p_ampa.densities.keys(), p_ampa.densities.values())

