from brainscapes.ontologies import atlases,parcellations,spaces
from brainscapes import atlas as bsa

atlas = bsa.Atlas()
atlas.select_parcellation_scheme(parcellations.CYTOARCHITECTONIC_MAPS)
icbm152space = spaces.MNI_152_ICBM_2009C_NONLINEAR_ASYMMETRIC
icbm_map = atlas.get_map(icbm152space)
print(icbm_map.shape)

