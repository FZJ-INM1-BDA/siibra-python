import siibra
from siibra.features.feature import SpatialFeature

spatial_features = [mod.modality() for mod in siibra.features.modalities if issubclass(mod._FEATURETYPE, SpatialFeature) ]