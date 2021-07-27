# Copyright 2018-2020 Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nibabel.affines import apply_affine
from numpy import linalg as npl
import numpy as np
from collections import defaultdict

from . import parcellations, spaces, features, logger
from .region import Region
from .features.feature import GlobalFeature
from .commons import create_key,MapType
from .config import ConfigurationRegistry
from .space import Space

VERSION_BLACKLIST_WORDS=["beta","rc","alpha"]

class Atlas:

    def __init__(self,identifier,name):
        # Setup an empty. Use _add_space and _add_parcellation to complete
        # the setup.
        self.name = name
        self.id = identifier
        self.key = create_key(name)

        # no parcellation initialized at construction
        self.parcellations = [] # add with _add_parcellation
        self.spaces = [] # add with _add_space

        # nothing selected yet at construction 
        self.selected_region = None
        self.selected_parcellation = None 

    def __hash__(self):
        """
        Used for caching functions taking atlas object as an input, like FeatureExtractor.pick_selection()
        """
        hash(self.id)+hash(self.selected_parcellation.id)+hash(self.selected_region)

    def _add_space(self, space):
        self.spaces.append(space)

    def _add_parcellation(self, parcellation, select=False):
        self.parcellations.append(parcellation)
        if self.selected_parcellation is None or select:

            self.select_parcellation(parcellation)

    def __str__(self):
        return self.name

    @staticmethod
    def from_json(obj):
        """
        Provides an object hook for the json library to construct an Atlas
        object from a json stream.
        """
        if all([ 
            '@id' in obj, 
            'spaces' in obj, 
            'parcellations' in obj ]):
            p = Atlas(obj['@id'], obj['name'])
            for space_id in obj['spaces']:
                if space_id not in spaces:
                    raise ValueError(f"Invalid atlas configuration for {str(p)} - space {space_id} not known")
                p._add_space( spaces[space_id] )
            for parcellation_id in obj['parcellations']:
                if parcellation_id not in parcellations:
                    raise ValueError(f"Invalid atlas configuration for {str(p)} - parcellation {parcellation_id} not known")
                p._add_parcellation( parcellations[parcellation_id] )
            return p
        return obj

    @property
    def regionnames(self):
        return self.selected_parcellation.names

    @property
    def regionlabels(self):
        return self.selected_parcellation.regiontree.labels


    def threshold_continuous_maps(self,threshold):
        """
        Inform the atlas that thresholded continuous maps should be preferred
        over static labelled maps for building and using region masks.
        This will, for example, influence spatial filtering of coordinate-based
        features in the get_features() method.
        """
        self.selected_parcellation.continuous_map_threshold = threshold

    def select(self,parcellation=None, region=None, force=False):
        """
        Select a parcellation and/or region. See Atlas.select_parcellation, Atlas.select_region
        """
        if parcellation is not None:
            self.select_parcellation(parcellation,force)
        if region is not None:
            self.select_region(region)
            
    def select_parcellation(self, parcellation,force=False):
        """
        Select a different parcellation for the atlas.

        Parameters
        ----------

        parcellation : Parcellation
            The new parcellation to be selected
        force : boolean
            If a parcellation is labelled "beta","rc" or "alpha", it cannot be selected unless force=True is provided as a positional
            argument.

        Yields
        ------
        Selected parcellation
        """
        parcellation_obj = parcellations[parcellation]
        if parcellation_obj.version is not None:
            versionname = parcellation_obj.version.name
            if any(w in versionname for w in VERSION_BLACKLIST_WORDS) and not force:
                logger.warning(f"Will not select experimental version {versionname} of {parcellation_obj.name} unless forced.")
                return
        if parcellation_obj not in self.parcellations:
            logger.error('The requested parcellation is not supported by the selected atlas.')
            logger.error('    Parcellation:  '+parcellation_obj.name)
            logger.error('    Atlas:         '+self.name)
            logger.error(parcellation_obj.id,self.parcellations)
            raise Exception('Invalid Parcellation')
        self.selected_parcellation = parcellation_obj
        self.selected_region = parcellation_obj.regiontree
        logger.info(f'Select "{self.selected_parcellation}"')
        return self.selected_parcellation

    def select_region(self,region):
        """
        Selects a particular region. 

        TODO test carefully for selections of branching points in the region
        hierarchy, then managing all regions under the tree. This is nontrivial
        because for incomplete parcellations, the union of all child regions
        might not represent the complete parent node in the hierarchy.

        Parameters
        ----------
        region : Region
            Region to be selected. Both a region object, as well as a region
            key (uppercase string identifier) are accepted.

        Yields
        ------
        Selected region
        """
        previous_selection = self.selected_region
        if isinstance(region,Region):
            # argument is already a region object - use it
            self.selected_region = region
        else:
            # try to interpret argument as the key for a region 
            selected = self.selected_parcellation.regiontree.find(
                    region,select_uppermost=True)
            if len(selected)==1:
                # one match found - fine
                self.selected_region = next(iter(selected))
            elif len(selected)==0:
                # no match found
                logger.error('Cannot select region. The spec "{}" does not match any known region.'.format(region))
            else:
                # multiple matches found. We do not allow this for now.
                logger.error('Cannot select region. The spec "{}" is not unique. It matches: {}'.format(
                    region,", ".join([s.name for s in selected])))
        if not self.selected_region == previous_selection:
            logger.info(f'Select "{self.selected_region.name}"')
        return self.selected_region

    def get_map(self, space=None, maptype=MapType.LABELLED):
        """
        return the map provided by the selected parcellation in the given space.
        This just forwards to the selected parcellation object, see
        Parcellation.get_map()
        """
        return self.selected_parcellation.get_map(space=space,maptype=maptype)

    def build_mask(self, space : Space, resolution_mm=None ):
        """
        Returns a binary mask in the given space, where nonzero values denote
        voxels corresponding to the current region selection of the atlas. 

        WARNING: Note that for selections of subtrees of the region hierarchy, this
        might include holes if the leaf regions are not completly covering
        their parent and the parent itself has no label index in the map.

        Parameters
        ----------
        space : Space
            Template space 
        resolution_mm : float or None (Default: None)
            Request the template at a particular physical resolution. If None,
            the native resolution is used.
            Currently, this only works for the BigBrain volume.
        """
        return self.selected_region.build_mask(space,resolution_mm=resolution_mm)

    def get_template(self,space=None):
        """
        Get the volumetric reference template image for the given space.

        See
        ---
        Space.get_template()

        Parameters
        ----------
        space : str
            Template space definition, given as a dictionary with an '@id' key

        Yields
        ------
        A nibabel Nifti object representing the reference template, or None if not available.
        TODO Returning None is not ideal, requires to implement a test on the other side. 
        """
        if space is None:
            space = self.spaces[0]
            if len(self.spaces)>1:
                logger.warning(f'{self.name} supports multiple spaces, but none was specified. Falling back to {space.name}.')


        try:
            spaceobj = spaces[space]
        except IndexError:
            logger.error(f'Atlas "{self.name}" does not support reference space "{space}".')
            print("Available spaces:")
            for space in self.spaces:
                print(space.name,space.id)
            return None

        return spaceobj.get_template()

    def decode_region(self,regionspec):
        """
        Given a unique specification, return the corresponding region from the selected parcellation.
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
        return self.selected_parcellation.decode_region(regionspec)

    def find_regions(self,regionspec,all_parcellations=False):
        """
        Find regions with the given specification in the current or all
        available parcellations of this atlas.

        Parameters
        ----------
        regionspec : any of 
            - a string with a possibly inexact name, which is matched both
              against the name and the identifier key, 
            - an integer, which is interpreted as a labelindex
            - a region object
        all_parcellations : Boolean (default:False)
            Do not only search the selected but instead all available parcellations.

        Yield
        -----
        list of matching regions
        """
        if not all_parcellations:
            return self.selected_parcellation.find_regions(regionspec)
        result = []
        for p in self.parcellations:
            result.extend(p.find_regions(regionspec))
        return result

    def clear_selection(self):
        """
        Cancels any current region selection.
        """
        self.select_region(self.selected_parcellation.regiontree)

    def region_selected(self,region):
        """
        Verifies wether a given region is part of the current selection.
        """
        return self.selected_region.includes(region)

    def coordinate_selected(self,space,coordinate):
        """
        Verifies wether a position in the given space is part of the current
        selection.

        Parameters
        ----------
        space : Space
            The template space in which the test shall be carried out
        coordinate : tuple x/y/z
            A coordinate position given in the physical space. It will be
            converted to the voxel space using the inverse affine matrix of the
            template space for the query.

        NOTE: since get_mask is lru-cached, this is not necessary slow
        """
        assert(space in self.spaces)
        # transform physical coordinates to voxel coordinates for the query
        def check(mask):
            voxel = (apply_affine(npl.inv(mask.affine),coordinate)+.5).astype(int)
            if np.any(voxel>=mask.dataobj.shape):
                return False
            if mask.dataobj[voxel[0],voxel[1],voxel[2]]==0:
                return False
            return True
        M = self.build_mask(space)
        return any(check(M.slicer[:,:,:,i]) for i in range(M.shape[3])) if M.ndim==4 else check(M)

    def get_features(self,modality=None,group_by_dataset=False,**kwargs):
        """
        Retrieve data features linked to the selected atlas configuration, by modality. 
        See siibra.features.modalities for available modalities. 
        """
        hits = []
        modalities = []

        if modality is None:
            modalities = features.modalities
        else:
            if modality not in features.modalities:
                logger.error("Cannot query features - no feature extractor known "\
                        "for feature type {}.".format(modality))
                return hits
            modalities = [modality]

        result = {}
        for m in modalities:
            for query in features.registry.queries(m,**kwargs):
                hits.extend(query.execute(self))
            matches = list(set(hits))
            if group_by_dataset:
                grouped = defaultdict(list)
                for m in matches:
                    grouped[m.dataset_id].append(matches)
                result[m]=grouped
            else:
                result[m]=matches
        
        # If only one modality was requested, simplify the dictionary
        if len(result)==1:
            return next(iter(result.values()))
        else:
            return result

    def assign_coordinates(self,space:Space,xyz_mm,sigma_mm=3):
        """
        Assign physical coordinates with optional standard deviation to atlas regions.
        See also: ContinuousParcellationMap.assign_coordinates()

        Parameters
        ----------
        space : Space
            reference template space for computing the assignemnt
        xyz_mm : coordinate tuple 
            3D point in physical coordinates of the template space of the
            ParcellationMap. Also accepts a string of the format "15.453mm, 4.828mm, 69.122mm" 
            as copied from siibra-explorer.
        sigma_mm : float (default: 0)
            standard deviation /expected localization accuracy of the point, in
            physical units. If nonzero, A 3D Gaussian distribution with that
            bandwidth will be used for representing the location instead of a
            deterministic coordinate.
        """
        smap = self.selected_parcellation.get_map(space,maptype=MapType.CONTINUOUS)
        return smap.assign_coordinates(xyz_mm, sigma_mm)

    def assign_maps(self,space:Space,mapimg):
        """
        Assign physical coordinates with optional standard deviation to atlas regions.
        See also: ContinuousParcellationMap.assign_coordinates()

        Parameters
        ----------
        space : Space
            reference template space for computing the assignemnt
        mapimg : 3D volume as nibabel spatial image
        """
        smap = self.selected_parcellation.get_map(space,maptype=MapType.CONTINUOUS)
        return smap.assign(mapimg)


REGISTRY = ConfigurationRegistry('atlases', Atlas)

if __name__ == '__main__':

    atlas = REGISTRY.MULTILEVEL_HUMAN_ATLAS

