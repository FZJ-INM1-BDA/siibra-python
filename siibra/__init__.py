# Copyright 2018-2020 Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .logging import LoggingContext,logger
QUIET = LoggingContext("ERROR")
VERBOSE = LoggingContext("DEBUG")

# __version__ is parsed by setup.py
__version__='0.1a8'
logger.info(f"Version: {__version__}")
logger.warning("This is a development release. Use at your own risk. Please file bugs and issues at https://github.com/FZJ-INM1-BDA/siibra-python.")

from .space import REGISTRY as spaces
from .parcellation import REGISTRY as parcellations
from .atlas import REGISTRY as atlases
from .features import modalities,gene_names
from .ebrains import KG_TOKEN as EBRAINS_KG_TOKEN
from .commons import MapType,ParcellationIndex
from .retrieval import Cache
CACHE = Cache.instance()

