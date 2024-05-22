from .cache import Cache, CACHE
from .warmup import WarmupLevel, WarmupParam, Warmup, WarmupRegException

from pathlib import Path

from joblib import Memory
jobmemory_path = Path(Cache.folder) / "joblib"
jobmemory_path.mkdir(parents=True, exist_ok=True)
jobmemory = Memory(jobmemory_path, verbose=0)
fn_call_cache = jobmemory.cache
