.. _examples:
==================
Examples
==================

This section provides a catalogue of documented code examples that is systematically organized along explain the core features of ``siibra``.

.. grid::

   .. grid-item-card:: :material-outlined:`account_tree;2em` Atlases and brain parcellations
      :link: examples/01_atlases_and_parcellations/index.html
      :link-type: url
      :columns: 12 12 12 12
      :class-card: sd-shadow-sm
      :margin: 2 2 auto auto

   .. grid-item-card:: :material-outlined:`map;2em` Maps and templates
      :link: examples/02_maps_and_templates/index.html
      :link-type: url
      :columns: 12 12 12 12
      :class-card: sd-shadow-sm
      :margin: 2 2 auto auto

   .. grid-item-card:: :octicon:`tasklist` Multimodal data features
      :link: examples/03_data_features/index.html
      :link-type: url
      :columns: 12 12 12 12
      :class-card: sd-shadow-sm
      :margin: 2 2 auto auto

   .. grid-item-card:: :material-outlined:`location_on;2em` Locations in reference spaces
      :link: examples/04_locations/index.html
      :link-type: url
      :columns: 12 12 12 12
      :class-card: sd-shadow-sm
      :margin: 2 2 auto auto

   .. grid-item-card:: :octicon:`pin` Anatomical assignment
      :link: examples/05_anatomical_assignment/index.html
      :link-type: url
      :columns: 12 12 12 12
      :class-card: sd-shadow-sm
      :margin: 2 2 auto auto

   .. grid-item-card:: :fas:`book` Detailed tutorials :octicon:`link-external`
      :link: https://github.com/FZJ-INM1-BDA/siibra-tutorials
      :link-type: url
      :columns: 12 12 12 12
      :class-card: sd-shadow-md
      :class-title: sd-text-primary
      :margin: 3 3 auto auto

      A set of jupyter notebooks demonstrating more extensive example use cases
      are maintained in the siibra-tutorials repository.


.. Note:: 
	
	Make sure you followed the steps in :ref:`getting-started`. Only very specialized functionalities require EBRAINS access which requires you to have an EBRAINS account and calling ``siibra.fetch_ebrains_token()`` to set the necessary token.
   
.. toctree::
   :hidden:
   :maxdepth: 3

   examples/01_atlases_and_parcellations/index
   examples/02_maps_and_templates/index
   examples/03_data_features/index
   examples/04_locations/index
   examples/05_anatomical_assignment/index

