<img align="right" src="https://github.com/FZJ-INM1-BDA/siibra-python/raw/main/images/siibra-python.jpeg" width="200">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/siibra.svg)](https://pypi.org/project/siibra/)
[![Python versions](https://img.shields.io/pypi/pyversions/siibra.svg)](https://pypi.python.org/pypi/siibra)
[![Documentation Status](https://readthedocs.org/projects/siibra-python/badge/?version=latest)](https://siibra-python.readthedocs.io/en/latest/?badge=latest)

# siibra - Python interface for interacting with brain atlases 

Copyright 2020-2021, Forschungszentrum Jülich GmbH 

*Authors: Big Data Analytics Group, Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH*

**Note: siibra-python is still at an experimental stage.** The API of the library is not stable, and the software is not yet fully tested. You are welcome to install and test it, but be aware that you will likely encounter bugs.


## Overview
<!--- Please keep this at line 19, or adjust the line skip in docs/readme.rst accordingly -->


`siibra` is a Python client for interacting with "multilevel" brain atlases, which combine multiple brain parcellations, reference coordinate spaces and modalities. 
`siibra` is designed to allow safe and convenient interaction with brain definitions from different parcellations, facilitating implementation of reproducible neuroscience workflows on the basis of brain atlases. It allows to work with reference brain templates both at millimeter and micrometer resolutions, and provides streamlined access to multimodal data features linked to brain regions. 

`siibra` is largely developed inside the [Human Brain Project](https://humanbrainproject.eu) for accessing the [EBRAINS human brain atlas](https://ebrains.eu/service/human-brain-atlas). 
It stores most of its configuration and data features in the [EBRAINS Knowledge Graph](https://kg.ebrains.eu), and is designed to support the [OpenMINDS metadata standards](https://github.com/HumanBrainProject/openMINDS_SANDS).

The functionality of `siibra-python` matches common actions known from browsing the interactive viewer `siibra explorer` [hosted on EBRAINS](https://atlases.ebrains.eu/viewer): 
Selecting different parcellations, browsing and searching brain region hierarchies, downloading maps, identifying brain regions, and accessing multimodal features and connectivity information associated with brain regions.

A key feature is a streamlined implementation of performing structured data queries for the main atlas concepts: reference spaces, parcellations, and brain regions. 
Accordingly, `siibra` implements unified handling for different types of features, namely

 - *spatial features* (which are linked to atlas regions via coordinates; like contact points of physiological electrodes), 
 - *regional features* (which are linked to atlases via a brain region specifications, like cell densities or neurotransmitter distributions), and 
 - *parcellation features* (linked to an atlas via a whole brain parcellation, like grouped connectivity matrices). 

As a result, all forms of data features can be queried using the same mechanism (`siibra.get_features()`) which accepts the specification of an concept (e.g. a selected brain region), and a data modality.
Currently available data features include neurotransmitter densities, regional connectivity profiles, connectivity matrices, high-resolution volumes of interest, gene expressions, and cell distributions. 
Additional features, including functional activation maps and electrophysiologal recordings, will become available soon.
Stay tuned!

`siibra` hides much of the complexity that would be required to interact with the individual data repositories that host the associated data.
By encapsulating many aspects of interacting with different maps and reference templates spaces, it also minimizes common errors like misinterpretation of coordinates from different reference spaces, mixing up label indices of brain regions, or utilisation of inconsistent versions of parcellation maps. 
It aims to provide a safe way of using maps defined across multiple spatial scales for reproducible analysis. 

## Documentation

`siibra-python`'s documentation is hosted at https://siibra-python.readthedocs.io.

## Usage examples

To get familiar with `siibra`, we recommend to checkout the jupyter notebooks in the `docs/` subfolder of the repository, which are the basis for much of the [documentation](https://siibra-python.readthedocs.io).


## Installation

### via pypi

`siibra` is available on pypi. To install the latest released version, simply run `pip install siibra`.

### via docker

```sh
# image is based on jupyter:scipy-notebook
docker run -dit \
    -p 10000:8888 \
    --rm \
    --name siibra \
    docker-registry.ebrains.eu/siibra/siibra-python:latest 
```

### via source

```sh
git clone https://github.com/FZJ-INM1-BDA/siibra-python.git
cd siibra-python
pip install .
```


## Setup

`siibra` retrieves much of its data from the [EBRAINS Knowledge Graph](https://kg.ebrains.eu), which requires authentication. 
Therefore you have to provide an EBRAINS authentication token for using all features provided by `siibra`.
Please make sure that you have a valid EBRAINS user account by [registering to EBRAINS](https://ebrains.eu/register/). 
Then follow the instructions for [obtaining EBRAINS API auth tokens](https://kg.ebrains.eu/develop.html).
As a last step, you need to fetch a recent token from the [authorization endpoint](https://nexus-iam.humanbrainproject.org/v0/oauth2/authorize), and make it known to `siibra` using either `siibra.set_ebrains_token()` or by storing it in the environment variable `HBP_AUTH_TOKEN`.  Note that as of now, you need to get a new token approximately every day to perform EBRAINS data queries. However, `siibra` implements a local cache on your harddisk, so once retrieved, your data will become usable and accessible without refreshing the token.


## Acknowledgements

This software code is funded from the European Union’s Horizon 2020 Framework
Programme for Research and Innovation under the Specific Grant Agreement No.
945539 (Human Brain Project SGA3).
