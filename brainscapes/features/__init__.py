from .feature import FeaturePool
from .receptors import ReceptorQuery as ReceptorPool
from .genes import AllenBrainAtlasQuery as GeneExpressionPool
from collections import defaultdict

pools_available = defaultdict(list)
for cls in FeaturePool.__subclasses__():
    pools_available[cls.__MODALITY__].append(cls)
