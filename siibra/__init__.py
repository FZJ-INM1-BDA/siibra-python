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

from os import path, getenv

# __version__ is parsed by setup.py
__version__='0.1a8'


# controlling the log level
import logging
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(name)s:%(levelname)s]  %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class LoggingContext:
    def __init__(self, level ):
        self.level = level
    def __enter__(self):
        self.old_level = logger.level
        logger.setLevel(self.level)
    def __exit__(self, et, ev, tb):
        logger.setLevel(self.old_level)

QUIET = LoggingContext("ERROR")
VERBOSE = LoggingContext("DEBUG")

def set_log_level(level):
    logger.setLevel(level)
set_log_level(getenv("SIIBRA_LOG_LEVEL","INFO"))

logger.info(f"Version: {__version__}")
logger.warning("This is a development release. Use at your own risk. Please file bugs and issues at https://github.com/FZJ-INM1-BDA/siibra-python.")

from .space import REGISTRY as spaces
from .parcellation import REGISTRY as parcellations
from .atlas import REGISTRY as atlases
from .retrieval import clear_cache
from .features import modalities
from .ebrains import set_token as set_ebrains_token
from .commons import MapType,ParcellationIndex

