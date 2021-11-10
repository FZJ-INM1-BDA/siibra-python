from typing import List
import siibra
from siibra.core import Atlas, Region, Space

# parameters

atlas_id=f'human'
space_id=f'mni152'
region_id=f'hoc1 left'

# code snippet

atlas: Atlas = siibra.atlases[f'{atlas_id}']
space: Space = atlas.spaces[f'{space_id}']
regions: List[Region] = atlas.find_regions(f'{region_id}')
region = regions[0]

centroids = region.centroids(space)
