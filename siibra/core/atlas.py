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

from .core import SemanticConcept
from .space import Space
from .parcellation import Parcellation
from .region import Region

from ..commons import MapType, logger
from ..features.query import FeatureQuery

from collections import defaultdict


VERSION_BLACKLIST_WORDS = ["beta", "rc", "alpha"]


class AtlasSelection:
    """
    Represents a particular region selection for the atlas,
    effectively defining the state of an atlas object.
    """

    def __init__(
        self, atlas, region: "Region", space: "Space" = None
    ) -> "AtlasSelection":
        """
        Create an atlas selection.

        Parameters
        ----------
        atlas : Atlas
            The source atlas object. It can be considered the context
            corresponding to the state defined by this object.
        parcellation: Parcellation
            The parcellation to select
        region : Region
            The selected region. If None, the root of the parcellation's
            region hierarchy will be chosen.
        space : Space
            The reference space to use.
        """
        self._atlas = atlas
        self.parcellation = region.parcellation
        self.region = region
        self.space = Space.REGISTRY[space]

    @property
    def regionnames(self):
        """Return the names of of the selected region and all its child regions."""
        return self.region.names

    @property
    def regionlabels(self):
        """Return the label indices of the selected region and all its child regions."""
        return self.region.regiontree.labels

    def __str__(self):
        if self.region == self.parcellation.regiontree:
            return f"'{self.parcellation.name}' in '{self.space.name}'"
        else:
            return f"Area '{self.region.name}' of '{self.parcellation.name}' in '{self.space.name}'"

    def get_map(self, maptype=MapType.LABELLED):
        """
        Return the map provided by the selected parcellation in the given space.
        This just forwards to the selected parcellation object, see
        Parcellation.get_map()

        Parameters
        ----------
        maptype : MapType
        """
        return self.parcellation.get_map(space=self.space, maptype=maptype)

    def get_regional_map(self, maptype=MapType.LABELLED):
        """
        Retrieves and returns a specific map of the selected region in the selected space,
        if available (otherwise None).

        Parameters
        ----------
        maptype : MapType
            Type of map (e.g. continuous, labelled - see commons.MapType)
        """
        return self.region.get_regional_map(self.space, maptype)

    def build_mask(self, resolution_mm=None):
        """
        Returns a binary mask in the given space, where nonzero values denote
        voxels corresponding to the current region selection of the atlas.

        WARNING: Note that for selections of subtrees of the region hierarchy, this
        might include holes if the leaf regions are not completly covering
        their parent and the parent itself has no label index in the map.

        Parameters
        ----------
        resolution_mm : float or None (Default: None)
            Request the template at a particular physical resolution. If None,
            the native resolution is used.
            Currently, this only works for the BigBrain volume.
        """
        return self.region.build_mask(self.space, resolution_mm=resolution_mm)

    def get_template(self):
        """
        Get the volumetric reference template image for the selected space.

        Parameters
        ----------
        resolution_mm : float or None (Default: None)
            Request the template at a particular physical resolution. If None,
            the native resolution is used.
            Currently, this only works for the BigBrain volume.

        Yields
        ------
        A nibabel Nifti object representing the reference template.
        """
        return self.space.get_template()

    def includes_region(self, region):
        """
        Verifies wether a given region is part of the selection,
        that is, a child of the selected region.
        """
        return self.region.includes(region)

    def get_features(self, modality=None, group_by_dataset=False, **kwargs):
        """
        Retrieve data features linked to the selected atlas configuration, by modality.
        See siibra.features.modalities for available modalities.
        """

        if modality is None:
            querytypes = [FeatureQuery.REGISTRY[m] for m in FeatureQuery.REGISTRY]
        else:
            if not FeatureQuery.REGISTRY.provides(modality):
                raise RuntimeError(
                    f"Cannot query features - no feature extractor known for feature type {modality}."
                )
            querytypes = [FeatureQuery.REGISTRY[modality]]

        result = {}
        for querytype in querytypes:
            hits = []
            for query in FeatureQuery.queries(querytype.modality(), **kwargs):
                hits.extend(query.execute(self))
            matches = list(set(hits))
            if group_by_dataset:
                grouped = defaultdict(list)
                for match in matches:
                    grouped[match.dataset_id].append(match)
                result[querytype.modality()] = grouped
            else:
                result[querytype.modality()] = matches

        # If only one modality was requested, simplify the dictionary
        if len(result) == 1:
            return next(iter(result.values()))
        else:
            return result

    def decode_region(self, regionspec):
        """
        Given a unique specification, return the corresponding region from this selection.
        The spec could be a label index, a (possibly incomplete) name, or a
        region object.
        This method is meant to definitely determine a valid region. Therefore,
        if no match is found, it raises a ValueError. If it finds multiple
        matches, it tries to return only the common parent node. If there are
        multiple remaining parent nodes, which is rare, a custom group region is constructed.

        Parameters
        ----------
        regionspec : any of
            - a string with a possibly inexact name, which is matched both
              against the name and the identifier key,
            - an integer, which is interpreted as a labelindex,
            - a region object

        Return
        ------
        Region object
        """
        return self.parcellation.decode_region(regionspec)

    def find_regions(self, regionspec):
        """
        Find regions with the given specification in the selected region.

        Parameters
        ----------
        regionspec : any of
            - a string with a possibly inexact name, which is matched both
              against the name and the identifier key,
            - an integer, which is interpreted as a labelindex
            - a region object

        Yield
        -----
        list of matching regions
        """
        return self.region.find_regions(regionspec)

    def assign_coordinates(self, xyz_mm, maptype=MapType.CONTINUOUS, sigma_mm=1):
        """
        Assign physical coordinates with optional standard deviation to atlas regions.
        See also: ContinuousParcellationMap.assign_coordinates()

        Parameters
        ----------
        xyz_mm : coordinate tuple
            3D point in physical coordinates of the template space of the
            ParcellationMap. Also accepts a string of the format "15.453mm, 4.828mm, 69.122mm"
            as copied from siibra-explorer.
        maptype : MapType
            wether to perform the assignment using continuous or labelled parcellation maps.
        sigma_mm : float (default: 0)
            standard deviation /expected localization accuracy of the point, in
            physical units. If nonzero, A 3D Gaussian distribution with that
            bandwidth will be used for representing the location instead of a
            deterministic coordinate.
        """
        smap = self.parcellation.get_map(self.space, maptype=maptype)
        return smap.assign_coordinates(xyz_mm, sigma_mm)

    def assign_maps(self, mapimg):
        """
        Assign physical coordinates with optional standard deviation to atlas regions.
        See also: ContinuousParcellationMap.assign_coordinates()

        Parameters
        ----------
        mapimg : 3D volume as nibabel spatial image
        """
        smap = self.selection.parcellation.get_map(
            self.space, maptype=MapType.CONTINUOUS
        )
        return smap.assign(mapimg)


@SemanticConcept.provide_registry
class Atlas(
    SemanticConcept, bootstrap_folder="atlases", type_id="juelich/iav/atlas/v1.0.0"
):
    """
    Main class for an atlas, providing access to feasible
    combinations of available parcellations and reference
    spaces, as well as common functionalities of those.
    """

    def __init__(self, identifier, name):
        """Construct an empty atlas object with a name and identifier."""

        SemanticConcept.__init__(self, identifier, name, dataset_specs=[])

        self.parcellations = []  # add with _add_parcellation
        self.spaces = []  # add with _add_space
        self.selection = None

    def __hash__(self):
        """
        Used for caching functions taking atlas object as an input, like FeatureExtractor.pick_selection()
        """
        (hash(self.id) + hash(self.selection.parcellation.id) + hash(self.selected_region))

    def _register_space(self, space):
        """Registers another reference space to the atlas."""
        self.spaces.append(space)

    def _register_parcellation(self, parcellation, select=False):
        """Registers another parcellation to the atlas."""
        self.parcellations.append(parcellation)

    @classmethod
    def _from_json(cls, obj):
        """
        Provides an object hook for the json library to construct an Atlas
        object from a json stream.
        """
        if obj.get("@type") != "juelich/iav/atlas/v1.0.0":
            raise ValueError(
                f"{cls.__name__} construction attempt from invalid json format (@type={obj.get('@type')}"
            )
        if all(["@id" in obj, "spaces" in obj, "parcellations" in obj]):
            atlas = cls(obj["@id"], obj["name"])
            for space_id in obj["spaces"]:
                if not Space.REGISTRY.provides(space_id):
                    raise ValueError(
                        f"Invalid atlas configuration for {str(atlas)} - space {space_id} not known"
                    )
                atlas._register_space(Space.REGISTRY[space_id])
            for parcellation_id in obj["parcellations"]:
                if not Parcellation.REGISTRY.provides(parcellation_id):
                    raise ValueError(
                        f"Invalid atlas configuration for {str(atlas)} - parcellation {parcellation_id} not known"
                    )
                atlas._register_parcellation(Parcellation.REGISTRY[parcellation_id])
            atlas.select()  # select defaults
            return atlas
        return obj

    def threshold_continuous_maps(self, threshold):
        """
        Inform the atlas that thresholded continuous maps should be preferred
        over static labelled maps for building and using region masks.
        This will, for example, influence spatial filtering of coordinate-based
        features in the get_features() method.
        """
        self.selection.parcellation.continuous_map_threshold = threshold

    def select(
        self,
        parcellation: Parcellation = None,
        region: "Region" = None,
        space: "Space" = None,
        allow_experimental=False,
    ) -> "AtlasSelection":
        """
        Modifies the selected region(s) of the atlas.

        Parameters
        ----------
        parcellation: Parcellation
            The parcellation to select. If None, the first of available parcellations is selected.
        region : Region
            Region to be selected. Both a region object, as well as a region
            key (uppercase string identifier) are accepted. If None,
            the root of the parcellation's region hierarchy will be chosen.
        space : Space
            The reference space to use. If None, the first of available spaces is selected.
        allow_experimental : bool
            Per default, experimental versions of parcellations will not
            be admitted for selection. Set to true to change this behaviour.
        """

        # clarify the default values
        if self.selection is None:
            default = AtlasSelection(
                self, self.parcellations[0].regiontree, self.spaces[0]
            )
        else:
            default = self.selection

        # determine the parcellation to select
        if parcellation is None:
            parcellation_obj = default.parcellation
        else:
            parcellation_obj = Parcellation.REGISTRY[parcellation]
            if parcellation_obj not in self.parcellations:
                raise ValueError(
                    f"Parcellation {parcellation_obj.name} not supported by atlas {self.name}."
                )

        if parcellation_obj.version is not None:
            versionname = parcellation_obj.version.name
            if (any(w in versionname for w in VERSION_BLACKLIST_WORDS) and not allow_experimental):
                logger.warning(
                    f"Will not select experimental version {versionname} of {parcellation_obj.name} unless forced."
                )
                return self.selection

        # determine the region to select
        if region is None:
            region_obj = default.region
            if region_obj.parcellation != parcellation_obj:
                logger.warning(
                    f"The previously selected region '{region_obj.name}' is not defined "
                    f"by '{parcellation_obj.name}'."
                )
                try:
                    corresponding_region = parcellation_obj.decode_region(
                        region_obj.name
                    )
                    region_obj = corresponding_region
                    logger.warning(
                        f"Selecting similar region {region_obj.name} of {parcellation_obj.name} instead."
                    )
                except Exception:
                    region_obj = parcellation_obj.regiontree
                    logger.warning(
                        f"Instead, the root of the region tree in '{parcellation_obj.name}' is selected."
                    )
        elif isinstance(region, Region):
            region_obj = region
        else:
            matches = parcellation_obj.regiontree.find(region, select_uppermost=True)
            if len(matches) == 0:
                raise ValueError(
                    'Cannot select region. The spec "{}" does not match any known region.'.format(
                        region
                    )
                )
            elif len(matches) == 1:
                region_obj = next(iter(matches))
            else:
                raise ValueError(
                    'Cannot select region. The spec "{}" is not unique. It matches: {}'.format(
                        region, ", ".join([s.name for s in matches])
                    )
                )

        if region_obj.parcellation != parcellation_obj:
            raise ValueError(
                f"Selected region {region_obj.name} is not defined in selected parcellation {parcellation_obj.name}."
            )

        # determine the space to select
        if space is None:
            space_obj = default.space
            if not parcellation_obj.supports_space(space_obj):
                space_obj = next(
                    iter(filter(parcellation_obj.supports_space, self.spaces))
                )
                logger.warning(
                    f"Previously selected space '{default.space.name}' is not supported "
                    f"by '{parcellation_obj.name}'. Selecting '{space_obj.name}' instead."
                )
        else:
            space_obj = Space.REGISTRY[space]
            if space_obj not in self.spaces:
                raise ValueError(
                    f"Space {space_obj.name} not supported by atlas {self.name}."
                )

        self.selection = AtlasSelection(parcellation_obj, region_obj, space_obj)
        logger.info(f"Selected {str(self.selection)}")
        return self.selection

    def find_regions(self, regionspec):
        """
        Find regions with the given specification in all
        parcellations offered by the atlas.

        Parameters
        ----------
        regionspec : any of
            - a string with a possibly inexact name, which is matched both
              against the name and the identifier key,
            - an integer, which is interpreted as a labelindex
            - a region object

        Yield
        -----
        list of matching regions
        """
        result = []
        for p in self.parcellations:
            result.extend(p.find_regions(regionspec))
        return result
