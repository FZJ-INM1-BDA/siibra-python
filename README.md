# Brainscapes client

Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.

## Installation

The library is not yet available over a global repository.
Installing for local usage: `pip install -e .`

## Authentication

Brainscapes retrieves data from the EBRAINS Knowledge Graph, which requires
authentication. You have to provide an authentication token for EBRAINS. 

Please obtain a token by visiting 
[https://nexus-iam.humanbrainproject.org/v0/oauth2/authorize](the EBRAINS authorization endpoint). 
Note that as of now, you have to get a new token
approximately every day.

There are two ways of making the token known to brainscapes:

1. Set an environment variable **HBP_AUTH_TOKEN** with the token. The client will then use it automatically.
2. Set it programmatically by getting an instance of **Authentication** as follows: 

```
authentication = Authentication.instance()
authentication.set_token(TOKEN)
```

## Usage

### Getting receptordata

```
from brainscapes.authentication import Authentication
from brainscapes.features import receptors
auth = Authentication.instance()
auth.set_token('eyJhbG..........')
print(receptors.get_receptor_data_by_region('Area 4p (PreCG)').fingerprint))
print(receptors.get_receptor_data_by_region('Area 4p (PreCG)').profiles)
print(receptors.get_receptor_data_by_region('Area 4p (PreCG)').autoradiographs)
```
