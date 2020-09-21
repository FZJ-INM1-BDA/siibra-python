import logging
logging.basicConfig(level=logging.INFO)

NAME2IDENTIFIER = lambda s : "".join(
        e if e.isalnum() else '_' 
        for e in s.strip()).upper()

def __compile_cachedir():
    from os import path,makedirs
    from appdirs import user_cache_dir
    cachedir = user_cache_dir(__name__,"")
    if not path.isdir(cachedir):
        makedirs(cachedir)
    return cachedir

CACHEDIR = __compile_cachedir()
logging.debug('Using cache: {}'.format(CACHEDIR))

from .definitions import spaces,parcellations,atlases
