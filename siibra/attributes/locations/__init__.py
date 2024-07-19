# Copyright 2018-2024
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

"""Handles spatial concepts and spatial operation like warping between spaces."""

from .location import Location
from .point import Point, Pt
from .pointset import PointSet, from_points, PointCloud
from .boundingbox import BoundingBox, BBox
from .polyline import Polyline
from .layerboundary import LayerBoundary
from .ops import intersect, union
from .base import Location as DataClsLocation