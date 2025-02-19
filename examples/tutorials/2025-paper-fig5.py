# %%
import siibra
assert siibra.__version__ >= "1.0.1"
import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting

# %%
# 1: get a region map
area, hemisphere = '4p', 'right'
parc = siibra.parcellations.get('julich 3.0.3')
region = parc.get_region(f"{area} {hemisphere}")
pmap = parc.get_map('mni152', 'statistical').get_volume(region)

# %%
# 2: find a corresponding brain section
patches = siibra.features.get(pmap, "BigBrain1MicronPatch", lower_threshold=0.8)

# %%
print(len(patches))

# %%
p1 = patches[0]
# %%
p1.fetch()
# %%
