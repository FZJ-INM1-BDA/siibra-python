# Adding new live queried AttributeCollection (Feature/Space/Parcellation/Map)

## Background

(TBD)

## Steps

1/ create a new fetcher:

```sh
$ touch my_query.py
```

2/ import your fetcher in module root

```diff
# in __init__.py

from . import bigbrain_profile
+ from . import my_query
```

3/ (optional)

Create modalit(ies) of interest, and register them to the modality vocabularies

```diff
# in my_query.py

+ from ...attributes.descriptions import Modality, register_modalities

+ my_modality = Modality(value="X-Ray Fluorescence Microscopy")
+ my_modality_2 = Modality(value="Multicellcular Spheroid")

+ @register_modalities()
+ def add_my_modalities():
+     yield my_modality
+     yield my_modality_2
```

This allows the modality to be autocompleted at runtime (for example, tab completion in interactive python session).

4/ register feature query

Lastly, we add the logic on how queries should be handled.

```diff
# in my_query.py
from ...attributes.descriptions import Modality, register_modalities
+ from ...concepts import Feature, AttributeCollection
+ from .base import LiveQuery

# trimmed for brevity

+ class MyLiveQuery(LiveQuery[Feature], generates=Feature):
+     def generate(self):
+         # do something with self.inputs
+         yield Feature()
```

## n.b.

It is worthwhile shortcircuiting and returning, to avoid long computations.
