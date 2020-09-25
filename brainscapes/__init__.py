import logging
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(name)s:%(levelname)s]  %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

from .space import REGISTRY as spaces
from .parcellation import REGISTRY as parcellations
from .atlas import REGISTRY as atlases

logger.setLevel(logging.INFO)
