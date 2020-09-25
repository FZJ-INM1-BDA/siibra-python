
def _build_registry():
    from .receptors import ReceptorQuery
    from .genes import AllenBrainAtlasQuery
    from .connectivity import ConnectivityQuery
    from .feature import FeaturePoolRegistry
    return FeaturePoolRegistry()
   
pools = _build_registry()
modalities = dir(pools)
