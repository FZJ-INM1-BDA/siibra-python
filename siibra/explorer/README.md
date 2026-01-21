# (experimental) Explorer extension

Control siibra-explorer with python

```python
from siibra.explorer.plugin import Explorer
explorer = Explorer()
explorer.start()

# Open the link specified in the console
# And click "OK" to open the companion plugin

explorer.navigate(position=(1e7,1e7,1e7)) # in nm

explorer.overlay(url="nifti://https://data-proxy.ebrains.eu/api/v1/public/buckets/d-d69b70e2-3002-4eaf-9c61-9c56f019bbc8/probabilistic_maps_pmaps_175areas/Area-hOc1/Area-hOc1_pmap_l_N10_nlin2ICBM152asym2009c_4.2_public_258e8c1d846f92be76922b20287344ae.nii.gz") # TODO a bit buggy, and does not yet work

from siibra.locations import PointSet
import siibra

ptst = PointSet([
    [1,2,3],
    [2,3,4]
], space=siibra.spaces['mni152'])

explorer.annotate(points=ptst) # maximize perspective view for best effect

```

## Automated tests

use [playwrite](https://playwright.dev/python/) to automate user interactions.

n.b. this functionality will depend largely on network speed, and geographical closeness to the siibra data centers.

note: despite stdout, do not interact with 

### Installation

```sh
pip install -r siibra/explorer/requirements.txt
playwright install chromium
```

### Example

see <sample_playwrite.py>

```sh
$ python siibra/explorer/sample_playwrite.py
<trimmed>
start Point in MNI Colin 27 [0.0,0.0,0.0]
navigated Point in MNI Colin 27 [10.0,10.0,10.0]
space_specced Point in MNI 152 ICBM 2009c Nonlinear Asymmetric [9.323640000000012,11.565300000000008,9.66219000000001]
returned Point in MNI Colin 27 [9.995550000000009,9.996489999999994,10.009999999999991]

```