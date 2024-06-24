from typing import TYPE_CHECKING, Set
from ..concepts import AtlasElement
from ..dataitems import IMAGE_FORMATS, IMAGE_VARIANT_KEY
from ..commons_new.iterable import get_ooo

if TYPE_CHECKING:
    from ..dataitems import Image


class Space(AtlasElement):
    schema: str = "siibra/atlases/space/v0.1"

    @property
    def images(self):
        from ..dataitems import Image

        return self._find(Image)

    @property
    def image_formats(self) -> Set[str]:
        formats = {im.format for im in self.images}
        if self.provides_mesh:
            formats = formats.union({"mesh"})
        if self.provides_volume:
            formats = formats.union({"volume"})
        return formats

    @property
    def variants(self):
        return {tmp.extra.get(IMAGE_VARIANT_KEY) for tmp in self.images} - {None}

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

    def get_template(self, frmt: str = None, variant: str = None) -> "Image":
        if frmt is None:
            frmt = [f for f in IMAGE_FORMATS if f in self.image_formats][0]
        else:
            assert (
                frmt in self.image_formats
            ), f"Requested format '{frmt}' is not available for this space: {self.image_formats=}."

        if variant:
            if not self.variants:
                raise ValueError("This space has no variants.")

            return get_ooo(
                self.images,
                lambda img: (img.format == frmt)
                and (variant.lower() in img.extra.get(IMAGE_VARIANT_KEY, "").lower()),
            )

        return get_ooo(self.images, lambda img: img.format == frmt)
