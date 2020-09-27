from ..commons import Glossary

def __init__():
    """
    Setup the module silently
    """
    from .receptors import ReceptorQuery
    from .genes import AllenBrainAtlasQuery
    from .connectivity import ConnectivityProfileExtractor, ConnectivityMatrixExtractor
    from .extractor import FeatureExtractorRegistry
    extractor_types = FeatureExtractorRegistry() 
    return [ extractor_types,
            Glossary(AllenBrainAtlasQuery.GENE_NAMES.keys()),
            Glossary(extractor_types.modalities.keys()) ]

extractor_types,gene_names,modalities = __init__()
classes = { name:extractor_types._extractors[name][0]._FEATURETYPE 
        for name in extractor_types._extractors.keys()}
