# Copyright 2018-2024
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .cache import Cache, CACHE
from .warmup import WarmupLevel, WarmupParam, Warmup, WarmupRegException

from pathlib import Path

from joblib import Memory
jobmemory_path = Path(Cache.folder) / "joblib"
jobmemory_path.mkdir(parents=True, exist_ok=True)
jobmemory = Memory(jobmemory_path, verbose=0)
fn_call_cache = jobmemory.cache
