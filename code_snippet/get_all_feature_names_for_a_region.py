import siibra
from siibra.features.feature import RegionalFeature

regional_feature_types = [mod for mod in siibra.features.modalities if issubclass(mod._FEATURETYPE, RegionalFeature) ]