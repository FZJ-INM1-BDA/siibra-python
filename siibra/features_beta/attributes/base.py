from dataclasses import dataclass, field

@dataclass
class Attribute:
    type: str = field(default="attr", init=False)
    key: str = field(default=None, init=False)
    value: str = field(default=None, init=False)

    def filter(self, *args, **kwargs):
        return True
