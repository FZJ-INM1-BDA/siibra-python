from dataclasses import dataclass
from typing import Iterable

from ..concepts import AtlasElement
from ..dataitems import Image, MESH_FORMATS, VOLUME_FORMATS, IMAGE_VARIANT_KEY


@dataclass
class Space(AtlasElement):
    schema: str = "siibra/atlases/space/v0.1"

    @property
    def images(self) -> Iterable[Image]:
        return self.get(Image)

    @property
    def meshes(self):
        return [image for image in self.images if image.format in MESH_FORMATS]

    @property
    def volumes(self):
        return [image for image in self.images if image.format in VOLUME_FORMATS]

    @property
    def provides_mesh(self):
        return len(self.meshes) > 0
    
    @property
    def provides_volume(self):
        return len(self.meshes) > 0
    
    def get_template(self, variant: str=None):
        for img in self.images:
            if variant and IMAGE_VARIANT_KEY in img.extra:
                if variant.lower() in img.extra[IMAGE_VARIANT_KEY].lower():
                    yield img
                continue
            yield img
