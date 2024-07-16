from typing import TYPE_CHECKING, Set, Union, List
from ..concepts import AtlasElement
from ..retrieval_new.volume_fetcher import (
    FetchKwargs,
    IMAGE_FORMATS,
    MESH_FORMATS,
)
from ..commons_new.iterable import assert_ooo
from ..commons_new.maps import merge_volumes
from ..dataitems import FORMAT_LOOKUP

if TYPE_CHECKING:
    from ..dataitems import Image, Mesh


class Space(AtlasElement):
    schema: str = "siibra/atlases/space/v0.1"

    @property
    def formats(self) -> Set[str]:
        return {vol.format for vol in self.volumes}

    @property
    def variants(self) -> List[str]:
        return list(
            dict.fromkeys(
                key
                for vol in self.volumes if vol.mapping is not None
                for key in vol.mapping.keys()
            )
        )

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

    def find_templates(
        self, variant: str = None, frmt: str = None
    ) -> List[Union["Image", "Mesh"]]:
        if frmt is None or frmt not in self.formats:
            frmt = [f for f in FORMAT_LOOKUP[frmt] if f in self.formats][0]
        else:
            assert frmt not in self.formats, RuntimeError(
                f"Requested format '{frmt}' is not available for this map: {self.formats=}."
            )

        if variant is not None:
            if self.variants:
                assert (
                    variant in self.variants
                ), f"{variant=!r} is not a valid variant for this space. Variants: {self.variants}"
            else:
                raise ValueError("This space has no variants.")

        def filter_templates(vol: Union["Image", "Mesh"]):
            if vol.mapping is None:
                return vol.format == frmt
            return vol.format == frmt and (
                (variant is None) or (variant in vol.mapping.keys())
            )

        return list(filter(filter_templates, self.volumes))

    def fetch_template(
        self,
        variant: str = None,
        frmt: str = None,
        **fetch_kwargs: FetchKwargs,
    ):
        if len(self.variants) > 1 and variant is None:
            _variant = self.variants[0]
            print(f"No variant was provided. Selecting the first of {self.variants=!r}")
        else:
            _variant = variant

        templates = self.find_templates(frmt=frmt, variant=_variant)
        if len(templates) == 0:
            raise ValueError(
                f"Could not get a template with provided parameters: ({variant=}, {frmt=})"
            )
        try:
            template = assert_ooo(templates)
            return template.fetch(**fetch_kwargs)
        except AssertionError:
            pass

        # check if only one variant has been selected and merge
        variants = dict.fromkeys(key for tmp in templates for key in tmp.mapping.keys())
        assert len(variants) == 1, f"Found several variants matching {_variant!r}. Please select a variant: {self.variants}"
        print("Found several volumes. Merging...")
        return merge_volumes([tmp.fetch(**fetch_kwargs) for tmp in templates])
