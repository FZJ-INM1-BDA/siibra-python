import logging
logging.basicConfig(level=logging.INFO)

from .space import REGISTRY as spaces
from .parcellation import REGISTRY as parcellations
from .atlas import REGISTRY as atlases
