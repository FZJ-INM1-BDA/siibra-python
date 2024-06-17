from dataclasses import dataclass
from typing import TYPE_CHECKING
from ..concepts import AtlasElement
from ..dataitems import IMAGE_FORMATS
from ..commons_new.iterable import get_ooo

if TYPE_CHECKING:
    from ..dataitems import Image


@dataclass
class Space(AtlasElement):
    schema: str = "siibra/atlases/space/v0.1"

    @property
    def images(self):
        from ..dataitems import Image

        return self._find(Image)

    @property
    def image_formats(self):
        return {img.format for img in self.images}

    @property
    def variants(self):
        return {tmp.extra.get("x-siibra/volume-variant") for tmp in self.images} - {
            None
        }

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

    def get_template(
        self, variant: str = None, frmt: str = None, fetch_kwargs=None
    ) -> "Image":
        if frmt is None:
            frmt = [f for f in IMAGE_FORMATS if f in self.image_formats][0]
        else:
            assert (
                frmt in self.image_formats
            ), f"Requested format '{frmt}' is not available for this space: {self.image_formats=}."

        if variant:
            if self.variants:
                images = [
                    img
                    for img in self.images
                    if variant.lower()
                    in img.extra.get("x-siibra/volume-variant", "").lower()
                ]
            else:
                import pdb

                pdb.set_trace()
                raise ValueError("This space has no variants.")
        else:
            images = self.images
        return get_ooo([img for img in images], lambda img: img.format == frmt)
