
from typing import Iterator, List
from abc import ABC, abstractmethod
from ...attributes import AttributeCollection


class LiveQuery(ABC):

    def __init__(self, input: List[AttributeCollection]):
        self._input = input

    def __init_subclass__(cls, generated_type):
        pass

    @abstractmethod
    def generate(self) -> Iterator[AttributeCollection]:
        # produces json specs of siibra concepts that matche the query
        pass
