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

import logging
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(name)s:%(levelname)s]  %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel("INFO")

# read in the package version from file
from os import path
PKG_DIR = path.dirname(path.abspath(__file__))
with open(path.join(PKG_DIR,"..","VERSION"),"r") as f:
    __version__ = f.read().strip()
logger.info("Version: "+__version__)

from .space import REGISTRY as spaces
from .parcellation import REGISTRY as parcellations
from .atlas import REGISTRY as atlases
from .retrieval import clear_cache
from .features import modalities

