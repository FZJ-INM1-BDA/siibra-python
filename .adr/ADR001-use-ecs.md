# ADR001: Adopt entity-component-system

## Status

Pending

## Context

In siibra-python v1, class of a feature is coupled to the modality. (e.g. `BlockfaceVolumeOfInterest` and `DTIVolumeOfInterest` only vary in `modality`/`category` attribute. In every other ways, they are identical). 

This tight coupling added a lot of inflexibility:

- This makes siibra-python code dependent on data. Adding new feature modalities (e.g. working with a local configuration directory), one must also update siibra-python code base.

- This creates unncessary overhead in class definitions, making stack track more difficult to resolve, code more difficult to maintain.

## Proposal

Adopt the entity-component-system[1] for not only features, but also other classes (spaces, maps, parcellations, regions). This allows the feature instance to be defined not by what it _is_, but what it _has_ (composition over inheritance). 

However, it is nice to have _some_ instance method to help with common tasks (e.g. `.plot` for features). 

As a compromise, we propose that we have the following classes:

```python
class Attribute: pass

class AttributeCollections:
    attributes: List[Attribute]

class Space(AttributeCollections): pass
class Region(AttributeCollections): pass
class Parcellation(Region): pass
class Map(AttributeCollections): pass
class Feature(AttributeCollections): pass
```

This proposal already drastically reduce the number of classes, and eliminates the tight coupling between data and code; whilst it still provides the flexibility of adding instance methods where necessary.

## Affected Design Philosophies

### extensibility (PRO)

This proposal should drastically improve extensibility:

- Features can now have arbitary modalities
- Features can now have multiple modalities. 
- Arbitary Attributes can be added, and be backwards compatible (progressive enhancement)

### usability/intuitiveness of API (CON)

The loss of OOP may mean that some methods are harder to find (e.g. `.boundary_positions`, `.boundary_annotation`)

We will need to find new homes for these (or escape hatch on how they can be calculated)

Perhaps we need a more generic/extensible scheme how different data pipelines can be modelled.


## Stakeholders

- Ahmet Simsek
- Xiao Gui

## References

[1] https://en.wikipedia.org/wiki/Entity_component_system

