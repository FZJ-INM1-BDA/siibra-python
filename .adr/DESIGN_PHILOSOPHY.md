# Design Philosophies & Design philosophy

This document is a summary of the design philosophies and architecture design outline. The architecture design should reference the relevant architecture design records.

## Design philosophies

This section outlines the highest level design philosphies for siibra-python. The points mentioned should be on the highest level and conceptual. Specific examples should be avoided. Subpoints may be provided to add context/clarification. 

The order represent the relative priorities of the philosophies. If an architecture decision would cause a conflict between two philosophies, this list should provide weighted assessment of the conflict, and the net positively/negative.

- siibra-land objects are context aware
    - are operable in siibra-land, annotated with rich metadata,
    - can be “exported” to interoperable-land to formats such as pandas dataframe/nifti etc, which lack rich metadata,
    <!-- - clearly define the scope of the “responsibility”. -->

- performant
    - User with low memory resource should be able to use siibra.
    - User with slow/intermittent internet connection should be able to (somewhat) use siibra.

- maintainability/debuggability

- extensibility/adaptability
    - siibra should be easily extendable (if user would like to add additional instances of features/maps/atlases etc.)
    - siibra should be easily adaptable in different domain (from brain to heart? to geology? to astronomy?)
    - developer/user should be able to extend the configuration with external repositories and APIs relatively easily

- portability/robustness
    - machine to machine, environment to environment
    - user’s OS should not affect their usage of siibra
    - user’s environment (e.g. may be air gapped) does not affect their usage of siibra

- analysis reproducibility
    - Same code should always produce the same result with the same version regardless of machine/env
    - If there is a random element involved, user should be able to set a seed

- usability/intuitiveness of the API

## Architecture design

foo bar [[ADR000]](ADR000-use-adr-records.md)