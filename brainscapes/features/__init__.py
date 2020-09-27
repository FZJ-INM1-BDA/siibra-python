from ..registry import Glossary

def __init__():
    """
    Setup the module silently
    """
    from .receptors import ReceptorQuery
    from .genes import AllenBrainAtlasQuery
    from .connectivity import ConnectivityProfileExtractor, ConnectivityMatrixExtractor
    from .feature import FeatureExtractorRegistry
    extractors = FeatureExtractorRegistry() 
    return [ extractors,
            Glossary(AllenBrainAtlasQuery.GENE_NAMES.keys()),
            Glossary(extractors.modalities.keys()) ]

extractors,gene_names,modalities = __init__()
classes = {name:extractors._extractors[name][0]._FEATURETYPE for name in extractors._extractors.keys()}
