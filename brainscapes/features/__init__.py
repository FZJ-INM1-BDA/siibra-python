from .genes import AllenBrainAtlasQuery

class Glossary:
    """
    A simple class that provides enum-like siple autocompletion for an
    arbitrary list of names.
    """

    def __init__(self,words):
        self.words = list(words)

    def __dir__(self):
        return self.words

    def __str__(self):
        return "\n".join(self.words)

    def __contains__(self,index):
        return index in self.__dir__()

    def __getattr__(self,name):
        if name in self.words:
            return name
        else:
            raise AttributeError("No such term: {}".format(name))


def __init__():
    from .receptors import ReceptorQuery
    from .genes import AllenBrainAtlasQuery
    from .connectivity import ConnectivityProfileParser, ConnectivityMatrixParser
    from .feature import FeaturePoolRegistry
    pools = FeaturePoolRegistry() 
    return [ pools,
            Glossary(AllenBrainAtlasQuery.GENE_NAMES.keys()),
            Glossary(pools.modalities.keys()) ]

pools,gene_names,modalities = __init__()
