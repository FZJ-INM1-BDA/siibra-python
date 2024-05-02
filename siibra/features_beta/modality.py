from ..commons import QUIET, KeyAccessor

_accessor: KeyAccessor = None

def __dir__():
    with QUIET:
        from .attributes.meta_attributes import ModalityAttribute
        modalities = ModalityAttribute._GetAll()
        global _accessor
        _accessor = KeyAccessor(names=list(modalities))
        return _accessor.__dir__()

def __getattr__(key: str):
    if _accessor is None:
        __dir__()
    return _accessor.__getattr__(key)
