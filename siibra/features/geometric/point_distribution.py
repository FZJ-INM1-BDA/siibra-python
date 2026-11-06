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
"""Distributions of points, such as cells, anchored to the regions they lie in."""

from typing import Dict, List, Sequence

import numpy as np

from .. import anchor as _anchor
from .. import feature
from ...core.assignment import Qualification
from ...core.region import Region
from ...geometries import points as _points
from ...geometries.providers.provider import GeometryProvider
from .geometric_feature import _GeometricFeature


class PointDistribution(
    _GeometricFeature,
    feature.Feature,
    _points.Points,
    configuration_folder="features/pointdistributions",
    category="generic",
):
    """
    Points measured in one subject, such as cell positions, with optional
    per-point attributes.

    The fragments are the points of one region each. `fetch(region=...)`
    loads the points of a region and its subregions, and
    `fetch(fragment=...)` loads fragments by name.
    """

    # Points sampled in a region describe that region.
    REGION_QUALIFICATION = Qualification.EXACT

    def __init__(
        self,
        modality: str,
        space_spec: dict,
        providers: List[GeometryProvider],
        anchor: _anchor.AnatomicalAnchor,
        name: str = "",
        description: str = "",
        subject: str = None,
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
            Specification of the space of the points, by "@id" or "name".
        providers: list of GeometryProvider
        anchor: AnatomicalAnchor
        name: str, default: ""
        description: str, default: ""
        subject: str, default: None
            The subject the points were measured in.
        fragment_regions: dict, default: None
            Regions each fragment lies in, by fragment name.
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
        _points.Points.__init__(
            self,
            space_spec=space_spec,
            providers=providers,
            name=name if subject is None else f"{name} - subject {subject}",
            affine=affine,
            fragment_regions=fragment_regions,
        )
        self.subject = subject
