# Architecture Design Summary

This document is a summary of the design philosophies and architecture design guidelines, distilled from the architecture design records (ADR). The summaries should be accompanied by the corresponding ADR(s), allowing convenient access to the full context when needed.

This document is aimed at those who are interested in the internal organisation of siibra-python codebase. When/if they are ready to contribute via pull requests, issues, feature rerquests etc, this document should be able to provide a more informed context.


## Design philosophies

This section outlines the highest level design philosphies for siibra-python. The points mentioned should be on the highest level and conceptual. Specific examples should be avoided. Subpoints may be provided to add context/clarification. 

The order represent the relative priorities of the philosophies. If an architecture decision would cause a conflict between two philosophies, this list should provide a weighted assessment of the conflict.

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

## Architecture design guidelines

### Summary table

| filename | Description |
|----------|---------|
| ADR000-use-adr-records.md | Employ architectural design records |
| ADR001-use-ecs.md | Core concepts are attribute collection (entity) composed of attributes (component) modelled after entity-component-system |
| ADR002-datarecipe.md | Data recipes (lazy loading with transparency) |
| | Schemas that describe siibra content and attributes |
| | Entity matching (including Point/Volume to parcellation map assignment) |
| | Factory (constructing content lazily) |
| | (To be discussed) caching strategies in terms of speed vs memory efficiency. choose default |
| | (To be discussed) handling uncertainty of locations and resolution |

### Data feature class design

Siibra data features should not defined by what they _are_, but by what they _have_ (i.e. do **not** extend `Feature` class). [[adr001]](ADR001-use-ecs.md)

```python
# n.b. simplified

class Attribute: pass

class AttributeCollections:
    attributes: List[Attribute]

class Space(AttributeCollections): pass
class Region(AttributeCollections): pass
class Parcellation(Region): pass
class Map(AttributeCollections): pass
class Feature(AttributeCollections): pass
```

> n.b. while `Space`, `Region`, `Parcellation` and `Map` is not necessarily relevant to the discussion of data feature design, they are included for completeness. 

