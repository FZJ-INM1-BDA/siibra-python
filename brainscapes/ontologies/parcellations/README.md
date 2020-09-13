The currently used formats have a few inconsistencies. In particular, the
Julich-Brain MPM map is provided as two Nifti files per template space, for
each hemisphere. Within the maps, they use the same labels. So V1 will have the
same label index in both maps. When computing spatial properties in the map,
this imposes a problem to assign them to the right region definition.

We have solved this here, by modeling multiple maps as a dictionary, where the
key to each map is a "description", which we expect to allow string matching
with the region names in the region hierarchy. In Julich-Brain, the two maps
are keyed by "left hemisphere" and "right hemisphere", which allows to match
each area to the correct region by string matching with the region name.

A future version should have a more robust implementation of the metadata scheme.

