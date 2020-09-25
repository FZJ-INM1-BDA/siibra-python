import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel('INFO')

from .space import REGISTRY as spaces
from .parcellation import REGISTRY as parcellations
from .atlas import REGISTRY as atlases
