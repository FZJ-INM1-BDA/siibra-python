import siibra
from siibra.features.feature import ParcellationFeature

parcellation_feature_types = [mod for mod in siibra.features.modalities if issubclass(mod._FEATURETYPE, ParcellationFeature) ]
