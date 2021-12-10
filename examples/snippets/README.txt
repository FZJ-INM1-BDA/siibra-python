Main features
=============

This section provides a catalogue of documented code examples that systematically walk you through the main features of `siibra`. 


.. topic:: Necessary preparations
	
	Before running the examples, make sure you followed the steps under "Getting started" in the `Introduction <https://siibra-python.readthedocs.io/en/latest/readme.html#getting-started>`_. In particular, you should have your EBRAINS account registered and enabled for API access in order to be able to generate access tokens.
	
While several of the functionalities do not require EBRAINS access, we will assume in the examples that you have set the environment variable `HBP_AUTH_TOKEN` to a freshly obtained EBRAINS access token. The token is a string sequence with more than 1000 characters, which usually begins with something like "eyJhbGciOiJSUzI1NiIsInR...". Instead of setting an environment variable, you can explicitely pass the token in your code:

::

	import siibra
	siibra.set_ebrains_token(<your token here>)

