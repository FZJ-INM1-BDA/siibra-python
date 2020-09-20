from . import genes

def __init__(): 

    from glob import glob
    from os import path
    import json

    globals()['sources'] = {}
    for folder in __path__:
        files = glob(path.join(folder,'*.json'))
        for fname in files:
            with open(fname,'r') as f:
                for k,v in json.load(f).items():
                    assert(k not in globals()['sources'].keys())
                    globals()['sources'][k] = v

__init__()
