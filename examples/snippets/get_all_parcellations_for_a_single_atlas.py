from typing import List
import siibra
from siibra.core import Atlas, Parcellation

# parameters

atlas_id=f'human'

# code snippet

atlas: Atlas = siibra.atlases[f'{atlas_id}']
parcellations: List[Parcellation] = [p for p in atlas.parcellations]