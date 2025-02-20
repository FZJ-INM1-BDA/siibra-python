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
region = parc.get_region("4p right")
pmap = parc.get_map('mni152', 'statistical').get_volume(region)

# %%
# 2: find a corresponding brain section
patches = siibra.features.get(pmap, "BigBrain1MicronPatch", lower_threshold=0.8)
print(f"Found {len(patches)} patches.")

# %%
N = 5
f, axs = plt.subplots(1, N)
for n, (patch, ax) in enumerate(zip(patches[:N], axs.ravel())):
    patchimg = patch.fetch()
    patchdata = patchimg.get_fdata().squeeze()
    ax.imshow(patchdata, cmap='gray', vmin=0, vmax=2**16)
    ax.axis('off')
    ax.set_title(f"Patch {n} - section {patch.bigbrain_section}")


# %%
