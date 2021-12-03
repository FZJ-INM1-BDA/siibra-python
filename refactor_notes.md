# Refactor notes

## Breaking changes

- `Region` no longer directly subclass NodeMixin. Instances has a `_node` attribute, which references the inherit `anytree` node
- `Region.has_parent` -> `Region.has_node_parent`
