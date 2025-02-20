# %%
import siibra
assert siibra.__version__ >= "1.0.1"
import matplotlib.pyplot as plt

# %%
# 1: Retrieve probability map of a motor area in Julich-Brain
parc = siibra.parcellations.get('julich 3.1')
region = parc.get_region("4p right")
pmap = parc.get_map('mni152', 'statistical').get_volume(region)

# %%
# 2: Extract BigBrain 1 micron patches with high probability in this area
patches = siibra.features.get(pmap, "BigBrain1MicronPatch", lower_threshold=0.7)
print(f"Found {len(patches)} patches.")

# %%
# 3: Display highly rated samples, here further reduced to a predefined section
section = 3556
candidates = filter(lambda p: p.bigbrain_section == 3556, patches)
f, axs = plt.subplots(1, 3, figsize=(8, 24))
for patch, ax in zip(list(candidates)[:3], axs.ravel()):
    patchdata = patch.fetch().get_fdata().squeeze()
    ax.imshow(patchdata, cmap='gray', vmin=0, vmax=2**16)
    ax.axis('off')
    ax.set_title(f"#{section} - {patch.vertex}", fontsize=10)


# %%
