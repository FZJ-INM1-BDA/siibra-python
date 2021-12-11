# Refactor notes

## Breaking changes

- `Region` no longer directly subclass NodeMixin. Instances has a `_node` attribute, which references the inherit `anytree` node
- `Region.has_parent` -> `Region.has_node_parent`
- `Region.find` has been removed. Falling back to `SiibraNode.find`
- `Parcellation.regiontree` now returns List of root regions instead of **a** root region
- Parcellation there is a unique parcellation for each parcellation/reference space combination (e.g. 6 parcellation instances for julich 29)
    - as a result, the `Atlas.get_parcellation` must be a bit more specific. e.g. `get_parcellation('2.9 mni 152')`
- `init_bootstrap` method added to openminds registry decorated class
    - import should not result in side effects
- rename a number of volume > NeuroglancerVolume attr (`{attrname}` -> `_{attrname}`)
- reworked `AtlasConcept.matches` method