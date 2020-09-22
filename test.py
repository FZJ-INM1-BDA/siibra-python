from brainscapes import atlases,parcellations,spaces

atlas = atlases[0]
atlas.select_parcellation(parcellations.JULICH_BRAIN_PROBABILISTIC_CYTOARCHITECTONIC_ATLAS)
icbm152space = spaces.MNI_152_ICBM_2009C_NONLINEAR_ASYMMETRIC
#icbm_map = atlas.get_map(icbm152space)
#print(icbm_map.shape)

print(atlas.regiontree)
matches = atlas.regiontree.find('hOc1',exact=False)
print(matches)


