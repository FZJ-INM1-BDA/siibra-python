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
"""Fiber bundles of a tractogram, anchored to the regions they terminate in."""

from typing import Dict, List, Sequence

import numpy as np

from .. import anchor as _anchor
from .. import feature
from ...core.assignment import Qualification
from ...core.region import Region
from ...geometries import tractogram as _tractogram
from ...geometries.providers.provider import GeometryProvider
from .geometric_feature import _GeometricFeature


class StreamlineFiberBundles(
    _GeometricFeature,
    feature.Feature,
    _tractogram.Tractogram,
    configuration_folder="features/connectivity/streamlinefiberbundles",
    category="connectivity",
):
    """
    The fiber bundles of a tractogram, anchored to the regions the bundles
    terminate in.

    The bundles are the fragments of the tractogram. `fetch(fragment=...)`
    loads bundles by id, and `fetch(region=...)` loads the bundles that
    terminate in a region or in any of its subregions.
    """

    # A bundle terminating in a region extends beyond it.
    REGION_QUALIFICATION = Qualification.OVERLAPS

    def __init__(
        self,
        modality: str,
        space_spec: dict,
        providers: List[GeometryProvider],
        anchor: _anchor.AnatomicalAnchor,
        name: str = "",
        description: str = "",
        fragment_regions: Dict[str, Sequence[Region]] = None,
        affine: np.ndarray = None,
        datasets: List = [],
        id: str = None,
        prerelease: bool = False,
    ):
        """
        Parameters
        ----------
        modality: str
        space_spec: dict
            Specification of the space of the streamlines, by "@id" or "name".
        providers: list of GeometryProvider
        anchor: AnatomicalAnchor
        name: str, default: ""
        description: str, default: ""
        fragment_regions: dict, default: None
            Regions each bundle terminates in, by bundle id.
        affine: numpy.ndarray, default: None
            4x4 affine mapping stored coordinates into the space, if needed.
        datasets: list, default: []
        id: str, default: None
        prerelease: bool, default: False
        """
        feature.Feature.__init__(
            self,
            modality=modality,
            description=description,
            anchor=anchor,
            datasets=datasets,
            id=id,
            prerelease=prerelease,
        )
        _tractogram.Tractogram.__init__(
            self,
            space_spec=space_spec,
            providers=providers,
            name=name,
            affine=affine,
            fragment_regions=fragment_regions,
        )
