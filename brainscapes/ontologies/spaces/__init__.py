from .. import *

# keep a reference to this module so that it's not garbage collected
old_module = sys.modules[__name__]

# setup the new module and patch it into the dict of loaded modules
sys.modules[__name__] = module(__name__)
