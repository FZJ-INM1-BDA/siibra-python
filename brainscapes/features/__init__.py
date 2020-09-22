from .feature import FeaturePool
from .receptors import ReceptorQuery as ReceptorPool
from .genes import AllenBrainAtlasQuery as GeneExpressionPool

def _build_registry():
    from .feature import FeaturePoolRegistry
    return FeaturePoolRegistry()
   
pools = _build_registry()
modalities = dir(pools)
