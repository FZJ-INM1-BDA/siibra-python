from abc import ABC, abstractmethod, abstractproperty

from pydantic import BaseModel
from pydantic.fields import Field

class AtAliasBaseModel(BaseModel):
    at_id: str = Field(alias='@id')
    at_type: str = Field(alias='@type')

class JSONableConcept(ABC):

    @property
    @abstractproperty
    def typed_json_output(self) -> BaseModel:
        pass

    @abstractmethod
    def to_json(self, detail=False, **kwargs):
        """
        Return a json dictionary representing this object.
        The key(s) must be of type str
        The value(s) must be of primitive type(s), or subclass of JSONableConcept:
        - str | int | float | JSONableConcept
        - (list|dict)[str|int|float|JSONableConcept]
        """
        pass
    
    @abstractmethod
    def from_json(self, **kwargs):
        pass