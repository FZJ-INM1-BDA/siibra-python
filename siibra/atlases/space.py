from dataclasses import dataclass

from ..concepts import AtlasElement


@dataclass
class Space(AtlasElement):
    schema: str = "siibra/atlases/space/v0.1"

    @property
    def images(self):
        from ..dataitems import Image
        return self._find(Image)

    @property
    def variants(self):
        from ..dataitems import Image
        return {tmp.variant for tmp in self.images if isinstance(tmp, Image)}

    @property
    def meshes(self):
        return [tmp for tmp in self.images if tmp.provides_mesh]

    @property
    def volumes(self):
        return [tmp for tmp in self.images if tmp.provides_volume]

    @property
    def provides_mesh(self):
        return len(self.meshes) > 0

    @property
    def provides_volume(self):
        return len(self.volumes) > 0

    def get_template(self, variant: str = None):
        if variant:
            for img in self.images:
                if variant.lower() in img.variant.lower():
                    return img

        else:
            pass
