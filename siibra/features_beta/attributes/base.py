from dataclasses import dataclass, field

SCHEMAS = {}


@dataclass
class Attribute:
    """ Base clase for attributes. """
    schema: str = field(default="siibra/attr", init=False)

    # derived classes set their schema as class parameter
    def __init_subclass__(cls, schema: str):
        cls.schema = schema
        SCHEMAS[schema] = cls

    def matches(self, *args, **kwargs):
        return False
