# ADR004: Entity matching

## Status

Pending

## Context

In different contexts, it become useful to access the relationship between two `AttributeCollection`s. Some examples may be:

- Given a `Region` instance, yield all related `Feature` instances

- For a specific `Feature` instance, yield all relationship qualifications to a `Region` instance

- Given a `Region` instance, yield all related `Region` instances

- For a specific `Region` instance, yield all relationship qualifications to a `Region` instance

The above questions are not trivial. For example:

- how can relationship(s) between `AttributeCollection`s be established?

- how can many degrees of separation should the relationship(s) between `AttributeCollection` be considered valid?

    - if `region_a` relates to `region_b`, `region_b` relates to `feature_b`, does `region_a` relate to `feature_b`?

    - if `region_a` relates to `feature_a`, `feature_a` relates to `feature_b`, does `region_a` relate to `feature_b`?

This design record is expected to:

- set expectations on the behavior of entity matching

- comprehensively address the PROs and CONs of the proposed approaches

## Proposal

### Matching subjects

Pertain to [adr001](ADR001-use.ecs.md), all entities that will be compared are instances of `AttributeCollection`. If, for whatever reason, an instance of `Attribute` needs to be matched, an _ad hoc_ `AttributeCollection` object with the said attribute should be constructed at runtime.

### Delegate to attribute matching

Pertain to [adr001](ADR001-use.ecs.md), an `AttributeCollection` is described by what it _has_, i.e., its `Attribute`s. Therefore, an efficient and logical reduction of the relationship between `AttributeCollection` is to access the relationship between the permutations of the respective `Attribute`s. This heuristic can be used to access relationships between any two instances of `AttributeCollection`.

### (out of scope) Quantification of attribute matching

Quantification of matches between `Attribute`s were not provided in siibra-python v1. While there may be some utilities, they are not considered in this ADR. Should the demand for attribute quantification arise in the future, this decision can be reconsidered.

### Qualitifcation of attribute matching

Qualification of matches between `Attribute`s should be represented with Enum.

```python

from enum import Enum

class Qualification(Enum):
    EXACT = 1
    OVERLAPS = 2
    CONTAINED = 3
    CONTAINS = 4
    APPROXIMATE = 5
    HOMOLOGOUS = 6
    OTHER_VERSION = 7

    @property
    def verb(self):
        """
        a string that can be used as a verb in a sentence
        for producing human-readable messages.
        """
        transl = {
            Qualification.EXACT: "coincides with",
            Qualification.OVERLAPS: "overlaps with",
            Qualification.CONTAINED: "is contained in",
            Qualification.CONTAINS: "contains",
            Qualification.APPROXIMATE: "approximates to",
            Qualification.HOMOLOGOUS: "is homologous to",
            Qualification.OTHER_VERSION: "is another version of",
        }
        assert self in transl, f"{str(self)} verb cannot be found."
        return transl[self]

    def invert(self):
        """
        Return qualification with the inverse meaning
        """
        inverses = {
            Qualification.EXACT: Qualification.EXACT,
            Qualification.OVERLAPS: Qualification.OVERLAPS,
            Qualification.CONTAINED: Qualification.CONTAINS,
            Qualification.CONTAINS: Qualification.CONTAINED,
            Qualification.APPROXIMATE: Qualification.APPROXIMATE,
            Qualification.HOMOLOGOUS: Qualification.HOMOLOGOUS,
            Qualification.OTHER_VERSION: Qualification.OTHER_VERSION,
        }
        assert self in inverses, f"{str(self)} inverses cannot be found."
        return inverses[self]
```


### (out of scope) Ranking of attribute match

Ranking the match results between `Attribute`s can provide some useful utility for sorting the respective `AttributeCollection`s based on relevance. Whilst it is clear that `Qualification.EXACT` should be ranked higher than the other instances of `Qualification`, ranking the remaining are not so clear. Partial implementation of ranking of `Qualification` can cause more confusion. Therefore, for now, this proposal is out of scope.

### Improve matching performance - register valid comparisons

On the surface, comparison of attributes creates a MxN problem. However, many of the comparisons can be skipped by explicitly declaring which `Attribute`s can be compared. e.g.

```python

REGISTRY = {}

def register_attribute_comparison(ClsA, ClsB):
    def outer(fn):
        REGISTRY[ClsA, ClsB] = fn, False
        REGISTRY[ClsB, ClsA] = fn, True
        return fn
    return outer

def compare_attribute(a, b):
    key = type(a), type(b)

    if key not in REGISTRY:
        return None
    fn, reverse_flag = REGISTRY[key]
    if reverse_flag:
        return fn(b, a)
    else:
        return fn(a, b)

@register_attribute_comparison(Point, Point)
def compare_pt_pt(pt: Point, pt: Point) -> Qualification: pass

@register_attribute_comparison(Point, PointCloud)
def compare_pt_ptcld(pt: Point, ptcld: PointCloud) -> Qualification: pass


assert type(test_a) is Point
assert type(test_b) is Point
assert type(test_c) is PointCloud
assert type(test_d) is BoundingBox
assert type(test_e) is Description

compare_attribute(test_a, test_b) # calls and returns compare_pt_pt
compare_attribute(test_a, test_c) # calls and returns compare_pt_ptcld
compare_attribute(test_a, test_d) # returns None, called neither
compare_attribute(test_a, test_e) # returns None, called neither
compare_attribute(test_d, test_e) # returns None, called neither
```

### Improve matching performance - lazy yield

To improve the robustness of the matching between `AttributeCollection`s, the qualification function should yield qualifcation result, allowing downstream consumers to decide for themselves if they would like to exhaust all qualifications or stop at the first one. e.g.

```python

def qualify_attribute_collections(
    ac_a: AttributeCollection, ac_b: AttributeCollection
) -> Iterator[Tuple[Attribute, Attribute, Qualification]]: pass

# to check if there is _at least_ one match
# optimising performance
def matches(
    ac_a: AttributeCollection, ac_b: AttributeCollection
) -> bool:
    try:
        if next(qualify_attribute_collections(col_a, col_b)):
            return True
        return False
    except StopIteration:
        return False

# to exhaustively get _all_ matches and their qualifications
# optimise completeness
def all_matches(
    ac_a: AttributeCollection, ac_b: AttributeCollection
) -> List[Tuple[Attribute, Attribute, Qualification]]:
    return list(qualify_attribute_collections(col_a, col_b))
```

### (out of scope) Further extensions - conditional matching

Whilst out of scope, some attribute matching may need to set partial flags, which will only match when the complimentary flag(s) are set.

One example of this is matching `Uberon` term (partial flag 1/2) and species (partial flag 2/2).

Here is one example how it may be implemented:

```python
from enum import Enum

# in addition to a comparison registry, we now also have a registry to populate partial flags
REGISTRY = {}
PARTIAL_REGISTRY = {}

# partial flags is its own special enum
class PartialFlags(Enum):
    UBERON_FLAG=1
    SPECIES_FLAG=2
    REGIONSPEC_FLAG=3

# we need to extend the existing attribute comparison to accomodate 
# the registration of partial flag comparisons
# we _can_ use a different registration function altogether
def register_attribute_comparison(ClsA, ClsB, *, partial_flag=False):
    def outer(fn):
        forward_key = ClsA, ClsB
        reverse_key = ClsB, ClsA
        if partial_flag:
            PARTIAL_REGISTRY[forward_key] = fn, False
            PARTIAL_REGISTRY[reverse_key] = fn, True
        else:
            REGISTRY[forward_key] = fn, False
            REGISTRY[reverse_key] = fn, True
        return fn
    return outer

# here are how partial flag registrations may look like
@register_attribute_comparison(UberonTerm, UberonTerm, partial_flag=True)
def compare_ub_ub(
    uberon_a: UberonTerm, uberon_b: UberonTerm
):
    return PartialFlags.UBERON_FLAG if uberon_a == uberon_b else None
@register_attribute_comparison(SpeciesSpec, SpeciesSpec, partial_flag=True)
def compare_species_species(
    species_a: SpeciesSpec, species_b: SpeciesSpec
):
    return PartialFlags.SPECIES_FLAG if species_a == species_b else None
@register_attribute_comparison(RegionSpec, RegionSpec, partial_flag=True)
def compare_species_species(
    regionspec_a: RegionSpec, region_spec_b: RegionSpec
):
    return PartialFlags.REGIONSPEC_FLAG if regionspec_a == region_spec_b else None

# here is one implementation how partial flag matching may be implemented

# we will need to define which partial flag combinations can result in a match
# this only needs to be done once on init
class Qualification(Enum):
    UBERON_SPECIES = 1024
    REGIONSPEC_SPECIES = 1025

PARTIAL_MATCH = {
    ({PartialFlags.UBERON_FLAG, PartialFlags.SPECIES_FLAG}): Qualification.UBERON_SPECIES,
    ({PartialFlags.REGIONSPEC_FLAG, PartialFlags.SPECIES_FLAG}): Qualification.REGIONSPEC_SPECIES
}

# at runtime, all flags will be accumulated ...
def aggregate_partial_flags(a: Attribute, b: Attribute, accumulator=None):
    if accumulator is None:
        accumulator = []
    key = type(a), type(b)
    if key not in PARTIAL_REGISTRY:
        return accumulator
    PARTIAL_REGISTRY[key] = fn, reverse_flag
    args = [b, a] if reverse_flag else [a, b]
    partial_flag = fn(*args)
    if partial_flag:
        return [*accumulator, partial_flag]
    else:
        return accumulator

# ... and checked all at once
from itertools import product
def qualify_attribute_collections(attr_coll_a: AttributeCollection, attr_coll_b: AttributeCollection):
    accumulator = []
    for attr_a, attr_b in product(attr_coll_a.attributes, attr_coll_b.attributes):
        accumulator = aggregate_partial_flags(attr_a, attr_b, accumulator)
    accumulator_set = set(accumulator)
    for key, value in PARTIAL_MATCH.items():
        if key in accumulator_set:
            yield accumulator_set[key]


```

## Affected Design Philosophies

### maintainability/debuggability (PRO)

This proposal reduces the the complex problem of assessing the relationship(s) between `AttributeCollection`s to multiple smaller, simpler, reusable, and managable problems of assessing relationship(s) between `Attribute`s. Because of the scope and functional nature of the assessment, unit tests can provide great coverage to prevent regression.

### performant (PRO)

The most basic implementation of this proposal allow for efficient matching (shortcircuit on first success).

### extensibility/adaptability (PRO)

The attribute matching mechanism is open for extension. An example of such extension is the [conditional matching](#out-of-scope-further-extensions---conditional-matching), while out of scope for the current ADR, will likely be introduced in the futre.

## Stakeholders

- Ahmet Simsek

- Xiao Gui

## References

