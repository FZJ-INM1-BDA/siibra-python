===========
How to cite
===========

Please cite the version of `siibra-python` used in your work. Version-specific
citation information is provided in the repository's `CITATION.cff` file and
through the Zenodo record associated with the software release.

Software citation
=================

The general software DOI for `siibra-python` is:

Timo Dickscheid, Xiaoyun Gui, Ahmet Nihat Simsek, Louisa Köhnen,
Vadim Marcenko, Christian Schiffer, Sebastian Bludau, and Katrin Amunts.
`siibra-python`. Zenodo. doi: 10.5281/zenodo.7885728

When possible, cite the specific release used for the analysis. This helps make
workflows reproducible and links results to the exact software version.

Tool-suite publication
======================

When referring to siibra as a software tool suite, the Multilevel Human Brain
Atlas, or the conceptual framework connecting atlases, reference spaces,
locations, and multimodal data features, cite the siibra tool-suite paper.

A peer-reviewed version of this manuscript has been accepted for publication in
Nature Methods and will be available soon. Until the final publication details
are available, please cite the preprint::

    Dickscheid, T., Gui, X., Simsek, A. N., Schiffer, C., Mangin, J.-F.,
    Leprince, Y., Jirsa, V., Bjaalie, J. G., Leergaard, T. B., Bludau, S.,
    and Amunts, K. Siibra: A software tool suite for realizing a Multilevel
    Human Brain Atlas from complex data resources. bioRxiv, 2025.
    doi: 10.1101/2025.05.20.655042


Citing atlas content and data features
======================================

`siibra-python` provides access to atlas maps, templates, and multimodal data
features that are often published as independent datasets. When a workflow uses
such content, cite the corresponding atlas or dataset publications in addition
to citing the software.

Relevant citations may include, depending on the workflow:

* `siibra-python` software release (``python -c "import siibra; print(siibra.__version__"``),
* parcellation, map, reference coordinate system, data feature (through `publications` attribute).
