# Copyright 2018-2026
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
"""Functional fingerprints of regions, cut from parcellation-wide activation tables."""

from typing import List

from . import query as _query
from ..features import tabular
from ..core import region as _region, parcellation as _parcellation


class FunctionalFingerprintQuery(
    _query.LiveQuery,
    args=[],
    FeatureType=tabular.FunctionalFingerprint,
):
    """
    For a region, finds the RegionalFunctionalActivation features of the region's
    own parcellation and returns one fingerprint per leaf of the region (the
    region itself if it is a leaf) that has data in it. Only that
    parcellation's file is downloaded; features of other parcellations,
    including other versions of the same parcellation, are never consulted.
    """

    def __init__(self, **kwargs):
        _query.LiveQuery.__init__(self, **kwargs)

    def query(self, concept, **kwargs) -> List[tabular.FunctionalFingerprint]:
        # Fingerprints are regional; whole parcellations and spaces are served
        # by RegionalFunctionalActivation itself.
        if not isinstance(concept, _region.Region) or isinstance(
            concept, _parcellation.Parcellation
        ):
            return []
        parcellation = concept.parcellation
        if parcellation is None:
            return []

        fingerprints = []
        for func_activation in tabular.RegionalFunctionalActivation.get_instances():
            if parcellation not in func_activation.anchor.represented_parcellations():
                continue
            data = func_activation.data
            with_data = {r for r in data.columns if data[r].notna().any()}
            fingerprints.extend(
                tabular.FunctionalFingerprint(region=leaf, func_activation=func_activation)
                for leaf in concept.leaves
                if leaf in with_data
            )
        return fingerprints
