from brainscapes.ontologies import atlases,parcellations,spaces
import brainscapes.human_multi_level_atlas as hba

atlas = hba.Atlas()
atlas.select_parcellation_scheme(parcellations.CYTOARCHITECTONIC_MAPS)
icbm152space = spaces.MNI_152_ICBM_2009C_NONLINEAR_ASYMMETRIC
icbm_map = atlas.get_map(icbm152space)
print(icbm_map.shape)

