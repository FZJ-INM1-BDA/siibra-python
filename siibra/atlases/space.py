# Copyright 2018-2024
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Set, Union, List
from ..concepts import AtlasElement
from ..attributes.dataproviders.volume import (
    VolumeOpsKwargs,
    IMAGE_FORMATS,
    MESH_FORMATS,
    FORMAT_LOOKUP,
)
from ..commons.iterable import assert_ooo
from ..commons.maps import merge_volumes

if TYPE_CHECKING:
    from ..attributes.dataproviders import ImageProvider, MeshProvider


class Space(AtlasElement):
    schema: str = "siibra/atlases/space/v0.1"

    @property
    def formats(self) -> Set[str]:
        return {vol.format for vol in self.volume_providers}

    @property
    def variants(self) -> List[str]:
        if self._attribute_mapping is None:
            return []
        return list(self._attribute_mapping.keys())

    @property
    def volume_providers(self):
        from ..attributes.dataproviders import ImageProvider, MeshProvider

        return [
            attr
            for attr in self.attributes
            if isinstance(attr, (MeshProvider, ImageProvider))
        ]

    @property
    def provides_mesh(self):
        return any(f in self.formats for f in MESH_FORMATS)

    @property
    def provides_image(self):
        return any(f in self.formats for f in IMAGE_FORMATS)

    def find_templates(
        self, variant: str = None, frmt: str = None
    ) -> List[Union["ImageProvider", "MeshProvider"]]:
        if frmt is None or frmt not in self.formats:
            frmt = [f for f in FORMAT_LOOKUP[frmt] if f in self.formats][0]
        else:
            assert frmt in self.formats, RuntimeError(
                f"Requested format '{frmt}' is not available for this space: {self.formats}."
            )

        if variant is not None:
            if self.variants:
                assert (
                    variant in self.variants
                ), f"{variant!r} is not a valid variant for this space. Variants: {self.variants}"
            else:
                raise ValueError("This space has no variants.")

        def filter_templates(vol: Union["ImageProvider", "MeshProvider"]):
            if len(self.variants) == 0:
                return vol.format == frmt
            return vol.format == frmt and (
                (variant is None) or (variant in self.variants)
            )

        return list(filter(filter_templates, self.volume_providers))

    def get_template(
        self,
        variant: Union[str, None] = None,
        frmt: str = None,
        **fetch_kwargs: VolumeOpsKwargs,
    ):
        if len(self.variants) > 1 and variant is None:
            _variant = self.variants[0]
            print(f"No variant was provided. Selecting the first of {self.variants!r}")
        else:
            _variant = variant

        templates = self.find_templates(frmt=frmt, variant=_variant)
        if len(templates) == 0:
            raise ValueError(
                f"Could not get a template with provided parameters: (variant{variant!r}, frmt={frmt})"
            )
        try:
            return assert_ooo(templates)
        except AssertionError:
            pass

        # check if only one variant has been selected and merge
        variants = set(tmp.name for tmp in templates) - {None}
        assert (
            len(variants) == 1
        ), f"Found several variants matching {_variant!r}. Please select a variant: {self.variants}"
        print("Found several volumes. Merging...")
        return merge_volumes([tmp.fetch(**fetch_kwargs) for tmp in templates])
