# Brainscapes 

## Installation

The library is not yet available over a global repository.
Installing for local usage: `pip install -e .`

## Authentication

Brainscapes retrieves some data from the EBRAINS Knowledge Graph, which requires
authentication. You have to provide an authentication token for EBRAINS. 

Please obtain a token by visiting 
[https://nexus-iam.humanbrainproject.org/v0/oauth2/authorize](the EBRAINS authorization endpoint). 
Note that as of now, you have to get a new token
approximately every day.

There are two ways of making the token known to brainscapes:

1. Set an environment variable **HBP_AUTH_TOKEN** with the token. The client will then use it automatically.
2. Set it programmatically by getting an instance of **Authentication** as follows: 

```python
from brainscapes.authentication import Authentication
auth = Authentication.instance()
auth.set_token(TOKEN)
```

## Usage examples

### Retrieving receptor densities for one brain area

```python
from brainscapes import atlases

# Retrieve data from atlas
# NOTE: assumes the client is already authenticated, see above
atlas = atlases.MULTILEVEL_HUMAN_ATLAS
for region in atlas.regiontree.find('hOc1',exact=False):
    atlas.select_region(region)
    hits = atlas.query_data("ReceptorDistribution")
    for hit in hits:
        print(hit)
```

### Retrieving gene expressions for one brain area

```python
from brainscapes import atlases

# Retrieve data from atlas
# NOTE: assumes the client is already authenticated, see above
atlas = atlases.MULTILEVEL_HUMAN_ATLAS
for region in atlas.regiontree.find('hOc1',exact=False):
    atlas.select_region(region)
    hits = atlas.query_data("GeneExpressions","GABAARL2")
    for hit in hits:
        print(hit)
```

## Command line interface

Many of the functionalities are available through the `brainscapes` commandline
client (brainscapes-cli). Note that many autocompletion functions are available
on the commandline, if you setup autompletion in your shell as described
[https://click.palletsprojects.com/en/7.x/bashcomplete/#](here).

Some examples:

 1. Retrieve receptor densities for a specific brain area:
	```brainscapes features AREA_HOC1__V1__17__CALCS_ receptors```
 2. Retrieve gene expressions for a specific brain area:
	```brainscapes features AREA_HOC1__V1__17__CALCS_ gex GABARAPL2```
 3. Print the region hierarchy:
    ```brainscapes hierarchy show```
 
