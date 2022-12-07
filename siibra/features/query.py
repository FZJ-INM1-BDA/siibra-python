from ..commons import logger

from abc import ABC, abstractmethod
from typing import List


class Query(ABC):

    # set of mandatory query argument names
    _query_args = []

    def __init__(self, **kwargs):
        parstr = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        if parstr:
            parstr = "with parameters " + parstr
        logger.info(f"Initializing query for {self._FEATURETYPE.__name__} features {parstr}")
        if not all(p in kwargs for p in self._query_args):
            raise ValueError(
                f"Incomplete specification for {self.__class__.__name__} query "
                f"(Mandatory arguments: {', '.join(self._query_args)})"
            )
        self._kwargs = kwargs

    def __init_subclass__(cls, args: List[str], objtype: type):
        cls._query_args = args
        cls.object_type = objtype
        return super().__init_subclass__()

    @abstractmethod
    def __iter__(self):
        """ iterate over queried objects (use yield to implemnet this in derived classes)"""
        pass

    @classmethod
    def get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass

    @classmethod
    def get_instances(cls, classname, **kwargs):
        # collect instances of the requested class from all suitable query subclasses.
        result = []
        for querytype in cls.get_subclasses():
            if querytype.object_type.__name__ == classname:
                result.extend(list(querytype(**kwargs)))
        return result
