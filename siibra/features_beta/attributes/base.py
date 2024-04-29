from dataclasses import dataclass, field

SCHEMAS = {}


@dataclass
class Attribute:
    """ Base clase for attributes. """
    schema: str = field(default="siibra/attr", init=False)

    # derived classes set their schema as class parameter
    def __init_subclass__(cls):
        assert cls.schema != Attribute.schema, f"Subclassed attributes must have unique schemas"
        assert cls.schema not in SCHEMAS, f"{cls.schema} already registered."
        SCHEMAS[cls.schema] = cls

    def matches(self, *args, **kwargs):
        return False
