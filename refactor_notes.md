# Refactor notes

## Breaking changes

- `Region` no longer directly subclass NodeMixin. Instances has a `_node` attribute, which references the inherit `anytree` node
- `Region.has_parent` -> `Region.has_node_parent`
- `Parcellation.regiontree` now returns List of root regions instead of **a** root region
- `init_bootstrap` method added to openminds registry decorated class
    - import should not result in side effects
- rename a number of volume > NeuroglancerVolume attr (`{attrname}` -> `_{attrname}`)