# (experimental) Explorer extension

```python
from siibra.explorer.plugin import Explorer
explorer = Explorer()
explorer.start()

# Open the link specified in the console
# And click "OK" to open the companion plugin

explorer.navigate(position=(1e7,1e7,1e7)) # in nm

explorer.overlay(url="nifti://https://data-proxy.ebrains.eu/api/v1/public/buckets/d-d69b70e2-3002-4eaf-9c61-9c56f019bbc8/probabilistic_maps_pmaps_175areas/Area-hOc1/Area-hOc1_pmap_l_N10_nlin2ICBM152asym2009c_4.2_public_258e8c1d846f92be76922b20287344ae.nii.gz") # TODO a bit buggy, and does not yet work

from siibra.locations import Point
import siibra

p1 = Point([1,2,3], space=siibra.spaces['mni152'])
p2 = Point([2,3,4], space=siibra.spaces['mni152'])
ps = p1.union(p2)

explorer.annotate(points=ps) # maximize perspective view for best effect

```