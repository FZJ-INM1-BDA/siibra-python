from ..commons import create_key, QUIET

MODALITY_DIC = {}

def __dir__():
    with QUIET:
        from .attributes.meta_attributes import ModalityAttribute
        modalities = ModalityAttribute._GetAll()
        global MODALITY_DIC
        MODALITY_DIC = {create_key(mod): mod for mod in modalities}
        return_val = list(MODALITY_DIC.keys())
        return_val.sort()
        return return_val

def __getattr__(key: str):
    return MODALITY_DIC[key]
