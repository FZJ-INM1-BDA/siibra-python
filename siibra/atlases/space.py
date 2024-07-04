from typing import TYPE_CHECKING, Set, Union
from ..concepts import AtlasElement
from ..retrieval_new.volume_fetcher import (
    FetchKwargs,
    IMAGE_FORMATS,
    MESH_FORMATS,
    VARIANT_KEY,
    FRAGMENT_KEY,
)
from ..commons_new.iterable import get_ooo

if TYPE_CHECKING:
    from ..dataitems import Image, Mesh


class Space(AtlasElement):
    schema: str = "siibra/atlases/space/v0.1"

    @property
    def formats(self) -> Set[str]:
        formats_ = {vol.format for vol in self.volumes}
        if any(f in IMAGE_FORMATS for f in formats_):
            formats_ = formats_.union({"image"})
        if any(f in MESH_FORMATS for f in formats_):
            formats_ = formats_.union({"mesh"})
        return formats_

    @property
    def variants(self):
        return {vol.extra.get(VARIANT_KEY) for vol in self.volumes} - {None}

    @property
    def fragments(self):
        return {vol.extra.get(FRAGMENT_KEY) for vol in self.volumes} - {None}

    @property
    def volumes(self):
        from ..dataitems import Image, Mesh

        return [attr for attr in self.attributes if isinstance(attr, (Mesh, Image))]

    @property
    def provides_mesh(self):
        return "mesh" in self.formats

    @property
    def provides_image(self):
        return "image" in self.formats

    def get_template(
        self, frmt: str = None, variant: str = None, fragment: str = None
    ) -> Union["Image", "Mesh"]:
        if frmt is None:
            frmt = [f for f in IMAGE_FORMATS + MESH_FORMATS if f in self.formats][0]
        else:
            assert (
                frmt in self.formats
            ), f"Requested format '{frmt}' is not available for this space: {self.formats=}."

        if variant:
            if not self.variants:
                raise ValueError("This space has no variants.")

            return get_ooo(
                self.volumes,
                lambda vol: (vol.format == frmt)
                and (variant.lower() in vol.extra.get(VARIANT_KEY, "").lower())
                and (fragment.lower() in vol.extra.get(FRAGMENT_KEY, "").lower()),
            )

        return get_ooo(self.volumes, lambda vol: vol.format == frmt)


def fetch_template(
    frmt: str = None,
    variant: str = None,
    fragment: str = None,
    **fetch_kwargs: FetchKwargs,
):
    # TODO: This can take one to interoprable formats so that siibra can automatically merge
    # fragments. Not sure `get_template` should do that. Maybe we rename get_template to find_templates
    # which returns either a list of Volumes or an attribute collection which we can fetch from.
    # needs discussion. Ideally, we should remove the need for FRAGMENT_KEY and VARIANT_KEY as well.
    raise NotImplementedError
