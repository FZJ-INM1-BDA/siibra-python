import brainscapes as bsc

# Setup the atlas
atlas = bsc.human_multi_level_atlas.Atlas()
jubrain = atlas.parcellations.CYTOARCHITECTONIC_MAPS
atlas.select_parcellation_schema(jubrain)

# Retrieve whole brain map in ICBM152 space as a NiftiImage
icbm152space = atlas.spaces.ICBM_152_2009C_NONL_ASYM
icbm_map = atlas.get_map(icbm152space)
print(icbm_map.shape)
print(icbm_map.header)
icbm_mri = atlas.get_template(icbm152space)
print(icbm_mri.shape)
print(icbm_mri.header)


# # For high resolution template spaces like BigBrain, a downscale factor or ROI
# # is required to retrieve a local file. Otherwise, a download URL and warning
# # message is returned
# bigbrain_map_400mu = atlas.get_template(spaces.spaces.BIGBRAIN_2015, resolution_mu=400)
region_v5 = atlas.get_region('Ch 123 (Basal Forebrain) - left hemisphere')
print(region_v5)
roi = region_v5.get_spatial_props(atlas.spaces.BIG_BRAIN)
print(roi)
# bigbrain_v1_map = atlas.get_template(spaces.spaces.BIGBRAIN_2015, roi=roi)

