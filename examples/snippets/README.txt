Main features
=============

This section provides a catalogue of documented code examples that systematically walk you through the main features of `siibra`. 


.. topic:: Necessary preparations
	
	Before running the examples, make sure you followed the steps under "Getting started" in the `Introduction <https://siibra-python.readthedocs.io/en/latest/readme.html#getting-started>`_. In particular, you should have your EBRAINS account registered and enabled for API access in order to be able to generate access tokens.
	While only some functionalities require EBRAINS access, the code examples assume that you set the environment variable `HBP_AUTH_TOKEN` to a freshly obtained EBRAINS access token. You can also pass it over via `siibra.set_ebrains_token(<your token here>)`. The token is a string sequence with more than 1000 characters usually starting with with "ey".

Before reading the examples, it helps to understand the main conceptual structures employed in `siibra`. These are:

 1. `Atlases`. An atlas in `siibra` is not a single brain map, but a collection of parcellation maps defined in one ore more reference spaces for a particular species. 
	→ :ref:`atlases`
 2. `Reference spaces`. A reference space defines a coordinate system in the brain. Reference spaces can be of different type (e.g. volumetric or surface-based, single-subject of average subject), and thus each atlas supports in general multiple reference spaces. Each reference space comes with at least one reference template, which is an image volume or mesh that represents the brain structures in that space.
	→ :ref:`parcmaps`
 3. `Parcellations`. A parcellation defines a (searchable) hierarchy of brain regions, as well as a set of available parcellation maps. Different parcellations may reflect different organizational principles of the brain, and thus an atlas can include support for multiple, typically complementary parcellations. 
	→ :ref:`parcellations`.
 4. `Parcellation maps`. A parcellation can be mapped in different reference spaces. Maps take the form of image volumes or surface meshes, and can be of 
	- "labelled" type, where coordinates or vertices in the reference space have a single unique brain region label, or of 
	- "continuous" type, where coordinates are linked to one floating point value per brain region, reflecting the weight or probability of each region at the given coordinate.
	→ :ref:`parcmaps`

Besides those core concepts, `siibra` defines structures for geometric primitives linked to reference spaces (like points and bounding boxes, → as well as structures for multimodal data features which are linked to brain locations or brain regions (→ :ref:`features`).

