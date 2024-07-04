from typing import TYPE_CHECKING, Set, Union, List
from ..concepts import AtlasElement
from ..retrieval_new.volume_fetcher import (
    FetchKwargs,
    IMAGE_FORMATS,
    MESH_FORMATS,
    VARIANT_KEY,
    FRAGMENT_KEY,
)
from ..commons_new.iterable import assert_ooo
from ..commons_new.maps import merge_volumes

if TYPE_CHECKING:
    from ..dataitems import Image, Mesh


class Space(AtlasElement):
    schema: str = "siibra/atlases/space/v0.1"

    @property
    def formats(self) -> Set[str]:
        return {vol.format for vol in self.volumes}

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
        return any(f in self.formats for f in MESH_FORMATS)

    @property
    def provides_image(self):
        return any(f in self.formats for f in IMAGE_FORMATS)

    def _find_templates(
        self, frmt: str = None, variant: str = "", fragment: str = ""
    ) -> List[Union["Image", "Mesh"]]:
        if frmt not in self.formats:  # TODO: this is repeated piece of code. see parcmap
            frmt_lookup = {
                None: IMAGE_FORMATS + MESH_FORMATS,
                "mesh": MESH_FORMATS,
                "image": IMAGE_FORMATS,
            }
            try:
                frmt = [f for f in frmt_lookup[frmt] if f in self.formats][0]
            except KeyError:
                raise RuntimeError(
                    f"Requested format '{frmt}' is not available for this map: {self.formats=}."
                )

        if variant and not self.variants:
            raise ValueError("This space has no variants.")
        if fragment and not self.fragments:
            raise ValueError("This space is not fragmented.")

        def filter_templates(vol: Union["Image", "Mesh"]):
            return (
                vol.format == frmt
                and (variant.lower() in vol.extra.get(VARIANT_KEY, "").lower())
                and (fragment.lower() in vol.extra.get(FRAGMENT_KEY, "").lower())
            )

        return list(filter(filter_templates, self.volumes))

    def fetch_template(
        self,
        frmt: str = None,
        variant: str = "",
        fragment: str = "",
        **fetch_kwargs: FetchKwargs,
    ):
        templates = self._find_templates(frmt=frmt, variant=variant, fragment=fragment)
        if len(templates) == 0:
            raise ValueError("Could not get a template with provided parameters.")
        try:
            template = assert_ooo(templates)
            return template.fetch(**fetch_kwargs)
        except AssertionError:
            pass
        # check if fragmented
        assert (
            len({tmp.extra.get(VARIANT_KEY, "") for tmp in templates}) == 1
        ), f"Found several variants matching {variant=}. Available variants: {self.variants}"
        assert (
            len({tmp.extra.get(FRAGMENT_KEY, "") for tmp in templates}) > 1
            and self.fragments
        ), "Found templates are not fragments of each other, cannot merge."
        return merge_volumes([tmp.fetch(**fetch_kwargs) for tmp in templates])
