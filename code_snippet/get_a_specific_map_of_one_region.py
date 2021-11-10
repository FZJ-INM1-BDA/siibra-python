from siibra.core import Atlas, Space, Parcellation, Region
import siibra
import nibabel as nib

# parameters

atlas_id = f'human'
parcellation_id = f'julich brain 2 9'
space_id = f'icbm 152'
region_id = f'hoc1 right'

# code snippet

atlas: Atlas = siibra.atlases[f'{atlas_id}']
parcellation: Parcellation = atlas.parcellations[f'{parcellation_id}']
space: Space = atlas.spaces[f'{space_id}']
region: Region = parcellation.find_regions(f'{region_id}')[0]

regional_map = region.get_regional_map(space, siibra.commons.MapType.CONTINUOUS)
nib.save(regional_map.image, 'continuous.nii.gz')