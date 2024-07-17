# Adding new live queried features

## Background

(TBD)

## Steps

1/ create a new fetcher:

```sh
$ touch my_fetcher.py
```

2/ import your fetcher in module root

```diff
# in __init__.py

from . import bigbrain_profile
+ from . import my_fetcher
```

3/ (optional)

Create modalit(ies) of interest, and register them to the modality vocabularies

```diff
# in my_fetcher.py

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
# in my_fetcher.py
from ...attributes.descriptions import Modality, register_modalities
+ from ...concepts import Feature, AttributeCollection
+ from ...assignment import register_collection_generator

# trimmed for brevity

+ @register_collection_generator(Feature)
+ def query_my_features(input: AttributeCollection):
+     import random
+     if random.random() > 0.5:
+         return
+     yield Feature()
```

## n.b.

It is worthwhile shortcircuiting and returning, to avoid long computations.