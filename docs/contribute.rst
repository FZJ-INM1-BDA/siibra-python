================
How to contribute
=================

Contributions to `siibra-python` are welcome through issue reports
(see :ref:`report_issues`), discussions, documentation improvements, and pull
requests.

The source code is hosted on GitHub: https://github.com/FZJ-INM1-BDA/siibra-python


Contributing code
=================

Code contributions can be proposed through pull requests. A focused pull
request is easier to review than a large change covering unrelated topics.

Before opening a pull request, check that the change is compatible with the
head of the branch and follows the existing package structure. Include or update
tests when the change affects behavior. Update documentation when public
behavior, recommended usage, or terminology changes.

For implementation-level information, see :ref:`developer`.

Contributing documentation
==========================

Documentation contributions are useful when they clarify concepts, fix outdated
usage patterns, improve examples, or make API behavior easier to understand.

When changing user-facing documentation, prefer the terminology used throughout
this documentation:

* reference atlas,
* terminology,
* brain area,
* reference coordinate system or space,
* annotation map,
* labelled map,
* statistical map,
* data feature,
* location or region of interest.

Adding atlas content or data features
=====================================

Atlas content and data features are not generally added by editing the
`siibra-python` source code directly. Foundational content is described in
configuration files, while dynamic content may be discovered through live
queries.

The documentation for adding and curating foundational content is maintained
separately:

:doc:`create_preconfiguration`
