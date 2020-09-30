# Copyright 2018-2020 Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
