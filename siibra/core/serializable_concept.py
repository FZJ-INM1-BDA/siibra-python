
from abc import ABC, abstractmethod


class JSONSerializable(ABC):

    @abstractmethod
    def to_model(self, **kwargs):
        raise AttributeError("JSONSerializable needs to have to_json overwritten")
