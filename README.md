# Brainscapes client

Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.

## Installation

The library is not yet available over a global repository.
Installing for local usage: `pip install -e .`

## Authentication

The client gets data from the Knowledge Graph and there for an authentication token must be set.
There are two ways of setting the token:

1. set an environment variable **HBP_AUTH_TOKEN** with the token, and the client will use it automatically
2. U can set it programmatically by getting an instance of **Authentication** 

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
