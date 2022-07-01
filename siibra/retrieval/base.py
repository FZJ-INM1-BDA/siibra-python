from abc import ABC, abstractmethod, abstractproperty

_NEEDS_TOBE_OVERRIDEN = "Needs to be overriden by subclasses"

class LazyLoader(ABC):

    @abstractmethod
    def get(self):
        raise NotImplementedError(_NEEDS_TOBE_OVERRIDEN)

    @abstractproperty
    def cached(self) -> bool:
        raise NotImplementedError(_NEEDS_TOBE_OVERRIDEN)

    @abstractproperty
    def data(self):
        raise NotImplementedError(_NEEDS_TOBE_OVERRIDEN)
