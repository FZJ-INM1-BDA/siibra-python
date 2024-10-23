import os
from contextlib import contextmanager

"""
since int/str are essentially non-mutable, defining the variables used
for configuration makes it difficult to patch at runtime (e.g. with context manager)
i.e. if developers do

```python
from siibra.commons.conf import KEEP_LOCAL_CACHE

if KEEP_LOCAL_CACHE:
    ...
```

... then it is basically impossible to update the KEEP_LOCAL_CACHE value.

To combat this, wrap the values in a class. Internal modules will need to

```python
from siibra.commons.conf import SiibraConf

if SiibraConf.KEEP_LOCAL_CACHE:
    ...
```
"""


# TODO: rename this class to avoid clash with configuration
class PerfConf:
    KEEP_LOCAL_CACHE = int(os.getenv("KEEP_LOCAL_CACHE", 0))
    MEMORY_HUNGRY = int(os.getenv("MEMORY_HUNGRY", 0))
    SIIBRA_MAX_FETCH_SIZE_GIB = float(os.getenv("SIIBRA_MAX_FETCH_SIZE_GIB", 0.2))

    @staticmethod
    @contextmanager
    def override_conf(keep_local_cache=None, memory_hungry=None, max_fetch_size=None):
        old_keep_local_cache, old_memory_hungry, old_fetch_gib = (
            PerfConf.KEEP_LOCAL_CACHE,
            PerfConf.MEMORY_HUNGRY,
            PerfConf.SIIBRA_MAX_FETCH_SIZE_GIB,
        )
        if keep_local_cache is not None:
            PerfConf.KEEP_LOCAL_CACHE = keep_local_cache
        if memory_hungry is not None:
            PerfConf.MEMORY_HUNGRY = memory_hungry
        if max_fetch_size is not None:
            PerfConf.SIIBRA_MAX_FETCH_SIZE_GIB = max_fetch_size
        yield
        (
            PerfConf.KEEP_LOCAL_CACHE,
            PerfConf.MEMORY_HUNGRY,
            PerfConf.SIIBRA_MAX_FETCH_SIZE_GIB,
        ) = (
            old_keep_local_cache,
            old_memory_hungry,
            old_fetch_gib,
        )
