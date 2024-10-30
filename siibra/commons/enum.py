from enum import EnumMeta, Enum


class _ContainedInEnumMeta(EnumMeta):

    def __contains__(cls, obj):
        try:
            cls(obj)
        except ValueError:
            return False
        return True


class ContainedInEnum(Enum, metaclass=_ContainedInEnumMeta):
    """Enum class that allow pythonic in checks"""

    pass
