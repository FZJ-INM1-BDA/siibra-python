siibra - Software interface for interacting with brain atlases
##############################################################

`siibra-python` is the Python interface of the siibra tool suite. It provides
programmatic access to reference atlases, brain-region terminologies, reference
coordinate systems, atlas maps, locations in the brain, and multimodal data
features. It links anatomical concepts with maps and measurements across
multiple spatial scales, from macroscopic MRI and surface spaces to microscopic
resources. The package supports scripts, notebooks, and reproducible analysis
workflows. Please see :ref:`glossary` page for details of the central concepts
used throughout the documentation.

With `siibra-python`, you can:

* browse and access brain areas, parcellations, spaces, templates, and maps,
* define coordinates, point sets, bounding boxes, and image-based regions of
  interest,
* assign locations and regions of interest to brain areas,
* query multimodal data features linked to brain areas or spatial locations,
* fetch image, tabular, and numerical data in Python-friendly formats,
* combine atlas information with common scientific Python tools,
* and combine these in workflows (see :ref:`usecases`).

`siibra-python` does not ship the atlas datasets themselves. Instead, it uses
metadata, configurations, and live queries to access distributed data resources
on demand. Data are fetched lazily and can be cached locally, so workflows only
download the elements they use.

.. grid::

  .. grid-item-card:: :fas:`rocket` Quickstart
    :link: getting-started
    :link-type: ref
    :columns: 12 12 6 6
    :class-card: sd-shadow-md
    :class-title: sd-text-primary
    :margin: 2 2 0 0

    Install ``siibra-python`` and run a minimal atlas query.

  .. grid-item-card:: :material-outlined:`stairs;1.5em` Examples
    :link: examples
    :link-type: ref
    :columns: 12 12 6 6
    :class-card: sd-shadow-md
    :class-title: sd-text-primary
    :margin: 2 2 0 0

    Browse documented code examples for common atlas workflows.

  .. grid-item-card:: :material-outlined:`psychology;1.5em` Use cases
    :link: usecases
    :link-type: ref
    :columns: 12 12 6 6
    :class-card: sd-shadow-md
    :class-title: sd-text-primary
    :margin: 2 2 0 0

    Explore neuroscience workflows ``siibra-python`` can be used in.

  .. grid-item-card:: :fas:`book` Tutorial repository :octicon:`link-external`
    :link: https://github.com/FZJ-INM1-BDA/siibra-tutorials
    :link-type: url
    :columns: 12 12 6 6
    :class-card: sd-shadow-md
    :class-title: sd-text-primary
    :margin: 2 2 0 0

    Tutorial notebooks for workshops and trainings with no-install options to run.

  .. grid-item-card:: siibra-explorer :octicon:`link-external`
    :link: https://atlases.ebrains.eu/viewer/
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-md
    :class-title: sd-text-primary
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
          :columns: 12 3 3 3

          .. image:: https://raw.githubusercontent.com/FZJ-INM1-BDA/siibra-explorer/refs/heads/master/docs/images/siibra-explorer-square.jpeg

      .. grid-item::
        :columns: 12 9 9 9

        .. div:: sd-font-weight-bold

        Use the web viewer to interactively browse reference spaces,
        templates, parcellation maps, brain areas, and linked multimodal
        data. It is a companion for discovering atlas content before using
        it in Python workflows.

  .. grid-item-card:: EBRAINS research infrastructure :octicon:`link-external`
    :link:  https://www.ebrains.eu/
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-md
    :class-title: sd-text-primary
    :margin: 2 2 auto auto
    
    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 3 3 3

        .. image:: https://search.kg.ebrains.eu/static/img/ebrains_logo.svg

      .. grid-item::
        :columns: 12 9 9 9

        .. div:: sd-font-weight-bold

        Many atlas elements and data features accessible through siibra are
        linked to resources hosted or registered in EBRAINS, supporting
        findable and reusable access to curated neuroscience data.

Acknowledgements
================

`siibra-python` is developed and maintained by the Big Data Analytics Group,
Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH,
with contributions from collaborators and users.

Development of `siibra-python` and the wider siibra tool suite has been
supported by European and national research infrastructure projects, including
the Human Brain Project and EBRAINS. The software contributes to atlas services
for accessing and using multilevel brain atlas content in reproducible
workflows.

We thank all contributors who have helped develop, test, document, and improve
`siibra-python` through code contributions, issue reports, discussions,
tutorials, workshops, and user feedback.

Many atlas elements and data features accessible through siibra are curated,
hosted, or registered through EBRAINS and related community resources. We
acknowledge the researchers, data providers, curators, and infrastructure teams
who make these resources available for reuse.

For details on software authorship and citation, see :doc:`howtocite`.


.. toctree::
  :hidden:
  :maxdepth: 2

  installation
  usecases
  concepts
  examples
  api
  faq
  howtocite

.. toctree::
  :hidden:
  :maxdepth: 2
  :caption: Development

  create_preconfiguration
  developer
  contribute
  report_issues
  GitHub Repository <https://github.com/FZJ-INM1-BDA/siibra-python>
