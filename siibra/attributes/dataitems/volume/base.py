from dataclasses import dataclass, field
from typing import Dict

from ..base import Data
from ....retrieval_new.volume_fetcher import IMAGE_FORMATS, MESH_FORMATS, Mapping


FORMAT_LOOKUP = {
    None: IMAGE_FORMATS + MESH_FORMATS,
    "mesh": MESH_FORMATS,
    "image": IMAGE_FORMATS,
}


@dataclass
class Volume(Data):
    schema: str = "siibra/attr/data/volume"
    space_id: str = None
    format: str = None
    url: str = None
    mapping: Dict[str, Mapping] = field(default=None, repr=False)
    colormap: str = field(default=None, repr=False)
