from ..registry import Glossary

def __init__():
    """
    Setup the module silently
    """
    from .receptors import ReceptorQuery
    from .genes import AllenBrainAtlasQuery
    from .connectivity import ConnectivityProfileParser, ConnectivityMatrixParser
    from .feature import FeaturePoolRegistry
    pools = FeaturePoolRegistry() 
    return [ pools,
            Glossary(AllenBrainAtlasQuery.GENE_NAMES.keys()),
            Glossary(pools.modalities.keys()) ]

pools,gene_names,modalities = __init__()
classes = {name:pools._pools[name][0]._FEATURETYPE for name in pools._pools.keys()}
