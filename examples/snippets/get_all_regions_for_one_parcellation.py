import siibra
from siibra.core import Parcellation, Atlas, Region

# parameters

atlas_id=f'human'
parcellation_id=f'julich brain 2 9'

# code snippet

atlas: Atlas = siibra.atlases[f'{atlas_id}']
parcellation: Parcellation = atlas.parcellations[f'{parcellation_id}']
regions: Region = parcellation.regiontree