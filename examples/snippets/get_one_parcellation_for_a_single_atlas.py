import siibra

# parameters

atlas_id=f'human'
parcellation_id=f'julich brain 2 9'

# code snippet

atlas = siibra.atlases[f'{atlas_id}']
parcellation = atlas.parcellations[f'{parcellation_id}']