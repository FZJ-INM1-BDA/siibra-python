import logging
from typing import Iterable, TypeVar
from tqdm import tqdm
from os.path import extsep

from ..commons import SIIBRA_LOG_LEVEL


logger = logging.getLogger(__name__.split(extsep)[0])
ch = logging.StreamHandler()
formatter = logging.Formatter("[{name}:{levelname}] {message}", style="{")
ch.setFormatter(formatter)
logger.addHandler(ch)


T = TypeVar("T")


def siibra_tqdm(iterable: Iterable[T] = None, *args, **kwargs):
    return tqdm(
        iterable,
        *args,
        disable=kwargs.pop("disable", False) or (logger.level > 20),
        **kwargs
    )


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
