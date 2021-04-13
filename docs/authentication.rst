==============
Authentication
==============

``siibra`` retrieves data from the `EBRAINS Knowledge Graph <https://kg.ebrains.eu>`_, which requires authentication. Therefore you have to provide an EBRAINS authentication token for using ``siibra``.

Please make sure that you have a valid EBRAINS user account by `registering to EBRAINS <https://ebrains.eu/register/>`_, and follow the `guidelines to get an EBRAINS API token <https://kg.ebrains.eu/develop.html>`_.
As a last step, you need to obtain an authentication token from the `authorization endpoint <https://nexus-iam.humanbrainproject.org/v0/oauth2/authorize>`_ and make it known to ``siibra``.
There are two ways to do so:

1. Set an environment variable **HBP_AUTH_TOKEN** with the token. The client will then use it automatically.
2. Set it programmatically by getting an instance of **Authentication** as follows: 
.. code:: python
    from siibra.authentication import Authentication
    auth = Authentication.instance()
    auth.set_token(TOKEN)

Note that as of now, you need to get a new token approximately every day to perform EBRAINS data queries. However, `siibra` implements a local cache on your harddisk, so once retrieved, your data will become usable and accessible without refreshing the token.

