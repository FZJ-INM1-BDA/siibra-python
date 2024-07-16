import logging
from os.path import extsep

from ..commons import SIIBRA_LOG_LEVEL

logger = logging.getLogger(__name__.split(extsep)[0])
ch = logging.StreamHandler()
formatter = logging.Formatter("[{name}:{levelname}] {message}", style="{")
ch.setFormatter(formatter)
logger.addHandler(ch)


class LoggingContext:
    def __init__(self, level):
        self.level = level

    def __enter__(self):
        self.old_level = logger.level
        logger.setLevel(self.level)

    def __exit__(self, et, ev, tb):
        logger.setLevel(self.old_level)


def set_log_level(level):
    logger.setLevel(level)


set_log_level(SIIBRA_LOG_LEVEL)
QUIET = LoggingContext("ERROR")
VERBOSE = LoggingContext("DEBUG")
