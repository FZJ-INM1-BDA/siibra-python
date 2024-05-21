from dataclasses import dataclass

from ..concepts.atlas_elements import AtlasElement
from ..descriptions.regionspec import RegionSpec
from ..exceptions import NotFoundException

@dataclass
class Region(
    AtlasElement,
    # anytree.NodeMixin
    ):
    schema: str = "siibra/attrCln/atlasEl/region"
