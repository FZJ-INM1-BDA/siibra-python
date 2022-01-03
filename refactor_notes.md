# Refactor notes

## TODO

- 01 / 005 -> get_regional_maps breaking change
- 02 / 001 -> alias with space
- 02 / 003 +

## Important conceptual changes

- Each parcellation (BrainAtlasVersion) **must** have an associated space. This means, for example, Julich 2.9 will need to be broken into 6 parcellations

## Breaking changes

- `Atlas.parcellations` order is no longer preserved. So it's no longer same to assume that the order parcellation appears in repo is a "default"
- `Region` no longer directly subclass NodeMixin. Instances has a `_node` attribute, which references the inherit `anytree` node
- `Region.has_parent` -> `Region.has_node_parent`
- `Region.find` has been removed. Falling back to `SiibraNode.find`
- ~~`Parcellation.regiontree` now returns List of root regions instead of **a** root region~~ This has been reworked to return **a** root region. One will be created if non exists.
- Parcellation there is a unique parcellation for each parcellation/reference space combination (e.g. 6 parcellation instances for julich 29)
    - as a result, the `Atlas.get_parcellation` must be a bit more specific. e.g. `get_parcellation('2.9 mni 152')`
- `init_bootstrap` method added to openminds registry decorated class
    - import should not result in side effects
- rename a number of volume > NeuroglancerVolume attr (`{attrname}` -> `_{attrname}`)
- reworked `AtlasConcept.matches` method
- `EbrainsDataset` migration:
    - no longer subclasses `Dataset`
        - attribute setter (`self.identifier`) does not work kindly with pydantic + openminds model
    - description has char limit of 3000
    - `ethics_assessment` cannot be easily mapped
    - `experiment_approach` cannot be easily mapped
    - `release_date` cannot easily be mapped. use 1970.1.1 as placeholder for now
    - `short_name` has max char length of 30
    - `technique` no easy way to map technique, using unknown for now
    - `version_innovation` -> placeholder