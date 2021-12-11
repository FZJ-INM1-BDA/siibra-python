Main features
=============

This section provides a catalogue of documented code examples, systematically organized along the core features of ``siibra``. 


.. attention:: 
	
	Make sure you followed the steps under "Getting started" in the `Introduction <https://siibra-python.readthedocs.io/en/latest/readme.html#getting-started>`_. You should have your EBRAINS account registered and enabled for API access in order to be able to generate access tokens. While only some functionalities require EBRAINS access, the code examples assume that you set the environment variable ``$HBP_AUTH_TOKEN`` to a freshly obtained EBRAINS access token. 


Before reading the examples, it helps to understand the main conceptual structures employed in ``siibra``:

 1. `Atlases`. An atlas in ``siibra`` can be understood as a collection of complementary parcellations and reference spaces at different spatial scales for a particular species, with functionality to access links between brain regions, spatial locations and anatomically organized data features. 
 2. `Reference spaces`. A reference space defines a coordinate system in the brain. Since reference spaces can be of different type (e.g. volumetric or surface-based, single-subject of average subject), an atlas can support multiple reference spaces. Each reference space comes with at least one reference template, which is an image volume or mesh representing the brain structures in that space.
 3. `Parcellations`. A parcellation defines a (searchable) hierarchy of brain regions and corresponding metadata, and thus represents a semantic object. Different parcellations may reflect different organizational principles of the brain, and thus an atlas can offer multiple, typically complementary parcellations. Each parcellation is linked to a set of corresponding parcellation maps.
 4. `Parcellation maps`. The regions defined by a given parcellation can be mapped in multiple reference spaces. The actual parcellation maps are spatial objects. They take the form of image volumes or surface meshes, and can be of **labelled type**, where coordinates or vertices in the reference space have a single unique brain region label, or of **continuous type**, where coordinates are linked to one floating point value per brain region, reflecting the weight or probability of each region at the given coordinate.
 5. `Regions`. A region represents a subtree of a region hierarchy, and contains metadata about the parcellation it belongs to, the parcellation maps where it is included, related publications, and more. Each region has links to its parent and child regions, if any.

Besides those core concepts, ``siibra`` also defines structures for geometric primitives linked to reference spaces (like points and bounding boxes), as well as structures for multimodal data features which are linked to brain locations or brain regions.

