from brainscapes import atlas as bsa
from brainscapes.ontologies import parcellations,spaces

# Setup the atlas
atlas = bsa.Atlas()
jubrain = parcellations.JULICH_BRAIN_PROBABILISTIC_CYTOARCHITECTONIC_ATLAS
atlas.select_parcellation_scheme(jubrain)

# Retrieve whole brain map in ICBM152 space as a NiftiImage
icbm152space = spaces.MNI_152_ICBM_2009C_NONLINEAR_ASYMMETRIC
for description,icbm_map in atlas.get_maps(icbm152space).items():
    print(description)
    print(icbm_map.shape)
    print(icbm_map.header)
icbm_mri = atlas.get_template(icbm152space)
print(icbm_mri.shape)
print(icbm_mri.header)

# # For high resolution template spaces like BigBrain, a downscale factor or ROI
# # is required to retrieve a local file. Otherwise, a download URL and warning
# # message is returned
bigbrain_map_400mu = atlas.get_template(spaces.spaces.BIGBRAIN_2015, resolution_mu=400)
# region_v5 = atlas.get_region('Ch 123 (Basal Forebrain) - left hemisphere')
# print(region_v5)
# roi = region_v5.get_spatial_props(spaces.BIG_BRAIN)
# print(roi)
# bigbrain_v1_map = atlas.get_template(spaces.spaces.BIGBRAIN_2015, roi=roi)
r1 = atlas.regions()[33]
print(r1)
print(r1.query_data(""))
