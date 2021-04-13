<p align="center">
    <img src="images/siibra-python.jpeg" width="450">
</p>

# siibra - Python interface for interacting with brain atlases 

*Authors: Big Data Analytics Group, Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH*

Copyright 2020-2021, Forschungszentrum Jülich GmbH 

> :warning: **`siibra-python` is still at an experimental stage.** The API of the library is not
stable, and the software is not yet fully tested. You are welcome to install and
test it, but be aware that you will likely encounter bugs.

## Overview

`siibra` is a Python client for interacting with "multilevel" brain atlases, which combine multiple brain parcellations and neuroscience datasets acrossdifferent reference template spaces. It is designed to allow safe and convenient interaction with brain regions from different parcellations, to provide streamlined access to multimodal data features linked to brain regions, and to perform basic analyses of region-specific data features. The intention of `siibra`  is to unify interaction with brain atlas data at different spatial scales, including parcellations and datasets at the millimeter scale in MNI space, as well as microstructural maps and microscopic data in the BigBrain space.

`siibra` is largely developed inside the [Human Brain Project](https://humanbrainproject.eu) for accessing the [human brain atlas of EBRAINS](https://ebrains.eu/service/human-brain-atlas). It retrieves most of its concept and datasets from the [EBRAINS Knowledge Graph](https://kg.ebrains.eu), and is designed to support the recently established [OpenMINDS metadata standards](https://github.com/HumanBrainProject/openMINDS_SANDS).

The functionality of `siibra` matches common actions known from browsing the [EBRAINS Interactive Atlas Viewer](https://atlases.ebrains.eu/viewer): Selecting different 
parcellations, browsing and searching brain region hierarchies, downloading maps, selecting regions, and accessing multimodal data features associated with brain regions.  A key feature is a streamlined implementation of performing structured data queries for selected brain regions, which gives access to multimodal regional “data features”. `siibra` implements a hierarchy of features, which unifies handling of *spatial data features* (which are linked to atlas regions via coordinates; like contact points of physiological electrodes), *regional data features* (which are linked to atlases via a brain region specifications, like cell densities or neurotransmitter distributions) and *global data features* (linked to an atlas via the whole brain or parcellation, like connectivity matrices or activation maps). As a result, all forms of data features can be queried using the same mechanism (`get_features()`) which takes as an argument a specification of the desired data modality, and respects the current selections made in the atlas. 
Currently, available data features include neurotransmitter densities, regional connectivity profiles, connectivity matrices, gene expressions, and spatial properties of brain regions. Additional features, including distributions of neuronal cells, functional activation maps and electrophysiologal recordings, will become available soon.

`siibra` hides much of the complexity that would be required to interact with the individual data repositories that host the associated data.
By encapsulating many aspects of interacting with different maps and reference templates spaces, it also minimizes common risks like misinterpretation of coordinates from different reference spaces, or utilisation of inconsistent versions of parcellation maps. 
It aims to provide a safe way of using maps defined across multiple spatial scales. 

## Documentation

The main place for `siibra-python`'s documentation is at https://siibra-python.readthedocs.io.

## Installation

`siibra` is available on pypi.
To install the latest version, simply run `pip install siibra`.

## Authentication

`siibra` retrieves data from the [EBRAINS Knowledge Graph](https://kg.ebrains.eu), which requires
authentication. Therefore you have to provide an EBRAINS authentication token for using `siibra`.

Please make sure that you have a valid EBRAINS user account by [registering to EBRAINS](https://ebrains.eu/register/), and follow the [guidelines to get an EBRAINS API token](https://kg.ebrains.eu/develop.html).
As a last step, you need to obtain an authentication token from the [authorization endpoint](https://nexus-iam.humanbrainproject.org/v0/oauth2/authorize) and make it known to `siibra`.
There are two ways to do so:

1. Set an environment variable **HBP_AUTH_TOKEN** with the token. The client will then use it automatically.
2. Set it programmatically by getting an instance of **Authentication** as follows: 
```python
from siibra.authentication import Authentication
auth = Authentication.instance()
auth.set_token(TOKEN)
```

Note that as of now, you need to get a new token approximately every day to
perform EBRAINS data queries. However, `siibra` implements a local cache on
your harddisk, so once retrieved, your data will become usable and accessible
without refreshing the token.

## Usage examples

To get familiar with `siibra`, we recommend to checkout the jupyter notebooks in the `examples/` subfolder of this repository. 
Below are some code snippets to give you an initial idea.

### Retrieving receptor densities for one brain area
```python
import siibra as sb
# NOTE: assumes the client is already authenticated, see above
atlas = sb.atlases.MULTILEVEL_HUMAN_ATLAS
atlas.select_region('v1')
features = atlas.get_features(
    sb.features.modalities.ReceptorDistribution)
for r in features:
    fig = r.plot(r.region)
```

### Retrieving gene expressions for one brain area

```python
import siibra as sb
from nilearn import plotting
atlas = sb.atlases.MULTILEVEL_HUMAN_ATLAS
# request gene expressions from Allen Atlas
atlas.select_region("v1 left")
features = atlas.get_features(
    sb.features.modalities.GeneExpression,
    gene=sb.features.gene_names.GABARAPL2 )
print(features[0])

# plot
all_coords = [tuple(g.location) for g in features]
mask = atlas.get_mask(sb.spaces.MNI_152_ICBM_2009C_NONLINEAR_ASYMMETRIC)
display = plotting.plot_roi(mask)
display.add_markers(all_coords,marker_size=5)
```

## Acknowledgements

This software code is funded from the European Union’s Horizon 2020 Framework
Programme for Research and Innovation under the Specific Grant Agreement No.
945539 (Human Brain Project SGA3).

