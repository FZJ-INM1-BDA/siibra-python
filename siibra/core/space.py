# Copyright 2018-2021
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
"""A particular brain reference space."""


from .concept import AtlasConcept

from ..commons import logger, Species

from typing import List, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..volumes import volume


class Space(AtlasConcept, configuration_folder="spaces"):

    def __init__(
        self,
        identifier: str,
        name: str,
        species: Union[str, Species],
        volumes: List['volume.Volume'] = [],
        shortname: str = "",
        description: str = "",
        modality: str = "",
        publications: list = [],
        datasets: list = [],
    ):
        """
        Constructs a new parcellation object.

        Parameters
        ----------
            identifier : str
                Unique identifier of the space
            name : str
                Human-readable name of the space
            species: str or Species
                Specification of the species
            volumes: list[Volume]
                list of template volumes
            shortname: str, optional
                Shortform of human-readable name
            description: str, optional
                Textual description of the parcellation
            modality  :  str or None
                Specification of the modality representing this reference space
            publications: list
                List of associated publications, each a dictionary with "doi" and/or "citation" fields
            ebrains_ids : dict
                Identifiers of EBRAINS entities corresponding to this Parcellation.
                Key: EBRAINS KG schema, value: EBRAINS KG @id
        """

        AtlasConcept.__init__(
            self,
            identifier=identifier,
            name=name,
            species=species,
            shortname=shortname,
            description=description,
            modality=modality,
            publications=publications,
            datasets=datasets,
        )
        self.volumes = volumes
        for v in self.volumes:
            v.space_info = {"@id": self.id}

    def get_template(self, variant: str = None):
        """
        Get the volumetric reference template for this space.

        Parameters
        ----------
            variant: str, optional
                Some templates are provided in different variants, e.g.
                freesurfer is available as either white matter, pial or
                inflated surface for left and right hemispheres (6 variants).
                This field could be used to request a specific variant.
                Per default, the first found variant is returned.

        Returns
        -------
            Volume
                representing the reference template, or None if not available.
        """
        tests = []
        if variant is not None:
            tests.append(lambda v: hasattr(v, 'variant') and variant.lower() in v.variant.lower())
        candidates = [v for v in self.volumes if all(t(v) for t in tests)]

        if len(candidates) == 0:
            msg = f"Volume variant {variant} not available for '{self.name}'. " \
                if variant else f"No volumes available for '{self.name}'. "
            raise RuntimeError(msg)

        if len(candidates) > 1:
            logger.info(
                f"Multiple template variants available for '{self.name}': "
                f"{', '.join(c.variant for c in candidates)}. "
                f"'{candidates[0].variant}' is chosen, but you might specify another with the 'variant' parameter."
            )

        template = candidates[0]
        if len(template.datasets) == 0:
            template.datasets = self.datasets
        return template

    @property
    def provides_mesh(self):
        return any(v.provides_mesh for v in self.volumes)

    @property
    def provides_image(self):
        return any(v.provides_image for v in self.volumes)
