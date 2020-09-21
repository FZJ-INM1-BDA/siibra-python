
import logging
logging.basicConfig(level=logging.INFO)

def __compile_cachedir():
    from os import path,makedirs
    from appdirs import user_cache_dir
    cachedir = user_cache_dir(__name__,"")
    if not path.isdir(cachedir):
        makedirs(cachedir)
    return cachedir

CACHEDIR = __compile_cachedir()
logging.debug('Using cache: {}'.format(CACHEDIR))
