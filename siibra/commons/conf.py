import os

KEEP_LOCAL_CACHE = int(os.getenv("KEEP_LOCAL_CACHE", 1))
MEMORY_HUNGRY = int(os.getenv("MEMORY_HUNGRY", 1))
SIIBRA_MAX_FETCH_SIZE_GIB = float(os.getenv("SIIBRA_MAX_FETCH_SIZE_GIB", 0.2))
