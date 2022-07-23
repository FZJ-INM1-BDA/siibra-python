# Copyright 2018-2021
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Want to include these queries with their features into the pre-populated lists
from .receptors import ReceptorDensityProfile, ReceptorFingerprint
from .voi import VolumeOfInterest
from .genes import AllenBrainAtlasQuery
from .connectivity import HcpStreamlineCountQuery, HcpRestingStateQuery, HcpStreamlineLengthQuery
from .connectivity import PrereleasedStreamlineLengthQuery, PrereleasedRestingStateQuery, PrereleasedStreamlineCountQuery
from .ebrains import EbrainsRegionalFeatureQuery
from .cells import RegionalCellDensityExtractor, CellDensityProfileQuery, CellDensityFingerprintQuery
from .ieeg import IEEG_SessionQuery
from .bigbrain import WagstylBigBrainProfileQuery
# from .morphologies import NeuroMorphoQuery

from .feature import Feature
get_features = Feature.get_features

__all__ = []
from .genes import GENE_NAMES as gene_names
modalities = Feature.get_modalities()
__all__ += ['gene_names', 'modalities']
