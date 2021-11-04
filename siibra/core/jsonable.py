from abc import ABC, abstractmethod, abstractproperty

from pydantic import BaseModel
from pydantic.fields import Field

class SiibraBaseSerialization(BaseModel):
    at_id: str = Field(alias='@id')
    at_type: str = Field(alias='@type')

class SiibraSerializable(ABC):

    @property
    @abstractproperty
    def SiibraSerializationSchema(self) -> BaseModel:
        pass

    @abstractmethod
    def to_json(self, detail=False, **kwargs):
        """
        Return a json dictionary representing this object.
        The key(s) must be of type str
        The value(s) must be of primitive type(s), or subclass of SiibraSerializable:
        - str | int | float | SiibraSerializable
        - (list|dict)[str|int|float|SiibraSerializable]
        """
        pass
    
    @abstractmethod
    def from_json(self, **kwargs):
        pass