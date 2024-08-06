# Migration guide from siibra v1 to v2

## Conceptual highlights

### `findter` vs `find` vs `get`

- `finditer_<foo>(spec)` will return an iterable of `<foo>` according to the `spec`

- `find_<foo>` will return a list of `<foo>` according to `spec` (proxy to `list(finditer_<foo>)`)

- `get_<foo>` will return the one and only instance of `<foo>`. Raise otherwise.

## Breaking Changes

- `siibra.features.get` -> `siibra.find_features`

- Since feature class hierarchy is removed, it is no longer to possible to query *all* features of a superclass (e.g. the following is will now raise: `siibra.find_features(parcellation, "RegionalConnectivity")`, and the following will have very different behavior: `siibra.find_features(parcellation, "connectivity")`)

- for `AllenGeneExpression` feature query, the keyword argument `genes` must be of type `List[str]` (previously, it was `Union[List[str], str]`)

