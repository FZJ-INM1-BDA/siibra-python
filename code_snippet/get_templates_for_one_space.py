from siibra.core import Atlas, Space
import siibra
import nibabel as nib

# parameters

atlas_id=f'human'
space_id=f'mni152'

# code snippet

atlas: Atlas = siibra.atlases[f'{atlas_id}']
space: Space = atlas.spaces[f'{space_id}']
nii = space.get_template()
nib.save(nii, 'template.nii.gz')