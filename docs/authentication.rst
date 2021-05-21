==============
Authentication
==============

``siibra`` retrieves data from the `EBRAINS Knowledge Graph <https://kg.ebrains.eu>`_, which requires authentication. Therefore you have to provide an EBRAINS authentication token for using ``siibra``.

Please make sure that you have a valid EBRAINS user account by `registering to EBRAINS <https://ebrains.eu/register/>`_, and follow the `guidelines to get an EBRAINS API token <https://kg.ebrains.eu/develop.html>`_.
As a last step, you need to obtain an authentication token from the `authorization endpoint <https://nexus-iam.humanbrainproject.org/v0/oauth2/authorize>`_. A nice way of doing this from a jupyter notebook is 

.. code:: python

    import webbrowser
    webbrowser.open('https://nexus-iam.humanbrainproject.org/v0/oauth2/authorize')
    token = input("Enter your token here: ")

Then you have to make the token known to ``siibra``.  There are two ways to do so:

1. Using the dedicated package method

   .. code:: python

   import siibra
   siibra.set_ebrains_token(token)

2. By setting the environment variable `HBP_AUTH_TOKEN` with the token. The client will then use it automatically. 

   .. code:: python

    from os import environ
    environ['HBP_AUTH_TOKEN'] = token

Note that as of now, you need to get a new token approximately every day to perform EBRAINS data queries. However, `siibra` implements a local cache on your harddisk, so once retrieved, data will become usable and accessible without refreshing the token.

