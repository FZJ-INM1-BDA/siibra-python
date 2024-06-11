from enum import Enum


class Qualification(Enum):
    EXACT = 1
    OVERLAPS = 2
    CONTAINED = 3
    CONTAINS = 4
    APPROXIMATE = 5
    HOMOLOGOUS = 6
    OTHER_VERSION = 7

    @property
    def verb(self):
        """
        a string that can be used as a verb in a sentence
        for producing human-readable messages.
        """
        transl = {
            Qualification.EXACT: "coincides with",
            Qualification.OVERLAPS: "overlaps with",
            Qualification.CONTAINED: "is contained in",
            Qualification.CONTAINS: "contains",
            Qualification.APPROXIMATE: "approximates to",
            Qualification.HOMOLOGOUS: "is homologous to",
            Qualification.OTHER_VERSION: "is another version of",
        }
        assert self in transl, f"{str(self)} verb cannot be found."
        return transl[self]

    def invert(self):
        """
        Return qualification with the inverse meaning
        """
        inverses = {
            Qualification.EXACT: Qualification.EXACT,
            Qualification.OVERLAPS: Qualification.OVERLAPS,
            Qualification.CONTAINED: Qualification.CONTAINS,
            Qualification.CONTAINS: Qualification.CONTAINED,
            Qualification.APPROXIMATE: Qualification.APPROXIMATE,
            Qualification.HOMOLOGOUS: Qualification.HOMOLOGOUS,
            Qualification.OTHER_VERSION: Qualification.OTHER_VERSION,
        }
        assert self in inverses, f"{str(self)} inverses cannot be found."
        return inverses[self]

    def __str__(self):
        return f"{self.__class__.__name__}={self.name.lower()}"

    def __repr__(self):
        return str(self)
