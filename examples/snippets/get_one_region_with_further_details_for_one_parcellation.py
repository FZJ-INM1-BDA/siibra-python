import siibra
from siibra.core import Parcellation, Atlas

# parameters

atlas_id=f'human'
parcellation_id=f'julich brain 2 9'
region_id=f'hoc1 right'

# code snippet

atlas: Atlas = siibra.atlases[f'{atlas_id}']
parcellation: Parcellation = atlas.parcellations[f'{parcellation_id}']
region = parcellation.find_regions(f'{region_id}')[0]