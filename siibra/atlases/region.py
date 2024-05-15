from dataclasses import dataclass

from ..concepts.atlas_elements import AtlasElement
from ..descriptions.regionspec import RegionSpec

@dataclass
class Region(
    AtlasElement,
    # anytree.NodeMixin
    ):
    schema: str = "siibra/attrCln/atlasEl/region"
    name: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.attributes.append(RegionSpec(value=self.name))
