from brainscapes_client import ebrains_humanbrainatlas as hba, parcellations, spaces

# Setup the atlas
atlas = hba.Atlas()
atlas.select_parcellation_schema(parcellations.JULICH_BRAIN_2_0)

# Retrieve whole brain map in ICBM152 space as a NiftiImage
icbm152space = spaces.ICBM_152_2009c_NONL_ASYM
icbm_map = atlas.get_map(icbm152space)
icbm_mri = atlas.get_template(icbm152space)

# For high resolution template spaces like BigBrain, a downscale factor or ROI
# is required to retrieve a local file. Otherwise, a download URL and warning
# message is returned
bigbrain_map_400mu = atlas.get_template(spaces.BIGBRAIN_2015, resolution_mu=400)
region_v5 = atlas.get_region('h0c5')
roi = region_v5.get_spatial_props(spaces.BIGBRAIN_2015).bounding_box
bigbrain_v1_map = atlas.get_template(spaces.BIGBRAIN_2015, roi=roi)

print(icbm152space)
