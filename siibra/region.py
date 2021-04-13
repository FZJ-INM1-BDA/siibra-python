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

from . import ebrains,logger
from .commons import create_key
from .retrieval import download_file 
from .space import Space
import numpy as np
import nibabel as nib
from nibabel.spatialimages import SpatialImage
import re
import anytree

class Region(anytree.NodeMixin):
    """
    Representation of a region with name and more optional attributes
    TODO implement a Region.from_json factory method to be applied recursively
    from the Parcellation.from_json method
    """

    def __init__(self, definition, parcellation, parent=None, children=None):
        """
        Constructs a region object from its definition as given in the
        siibra parcellation configurations.

        Parameters
        ----------
        definition : dict
            A dictionary of one particular region as formatted in the siibra parcellation defininition json files.
        parcellation : Parcellation
            The parcellation that this region belongs to
        parent : Region
            Parent of this region, if any
        children : list of Region
            Children of this region, if any
        """
        self.name = definition['name']
        self.key = create_key(self.name)
        self.parcellation = parcellation
        self.labelindex = None
        if 'labelIndex' in definition.keys():
            self.labelindex = definition['labelIndex'] 
        self.attrs = definition
        if parent is not None:
            self.parent = parent
        if children is not None: 
            self.children = children

    def _related_ebrains_files(self):
        """
        Returns a list of downloadable files from EBRAINS that could be found
        for this region, if any.
        FIXME: parameter is not used!
        """
        files = []
        if 'originDatasets' not in self.attrs.keys():
            return files
        if len(self.attrs['originDatasets'])==0:
            return files
        if 'kgId' not in self.attrs['originDatasets'][0].keys():
            return files
        dataset = self.attrs['originDatasets'][0]['kgId']
        res = ebrains.execute_query_by_id(
                'minds','core','dataset','v1.0.0',
                'brainscapes_files_in_dataset',
                params={'dataset':dataset} )
        for dataset in res['results']:
            files.extend(dataset['files'])
        return files

    def __eq__(self,other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        """
        Identify each region by its parcellation and region key.
        """
        return hash(self.parcellation.key+self.key)

    def has_parent(self,parentname):
        return parentname in [a.name for a in self.ancestors]

    def __getattr__(self,name):
        if name in self.attrs.keys():
            return self.attrs[name]
        else:
            raise AttributeError("No such attribute: {}".format(name))

    def includes(self, region):
        """
        Determine wether this regiontree includes the given region.
        """
        return region==self or region in self.descendants

    def find(self,name,select_uppermost=False):
        """
        Find region with the given name in all descendants of this region.

        Parameters
        ----------
        name : str
            The name to search for.
        select_uppermost : Boolean
            If true, only the uppermost matches in the region hierarchy are
            returned (otherwise all siblings as well if they match the name)

        Yield
        -----
        list of matching regions
        """
        result = anytree.search.findall(self,
                lambda node: node.might_be(name))
        if len(result)>1 and select_uppermost:
            all_results = result
            mindepth = min([r.depth for r in result])
            result = [r for r in all_results if r.depth==mindepth]
            if len(result)<len(all_results):
                logger.info("Using only {} parent nodes of in total {} matching regions for spec '{}'.".format(
                    len(result), len(all_results), name))
        if isinstance(result,Region):
            return {result}
        elif result is None:
            return set()
        else:
            return set(result)

    def might_be(self,regionspec):
        """ 
        Checks wether this region might match the given specification, which
        could be any of 
            - a string with a name,
            - an integer, interpreted as a labelIndex,
            - a region object
        """
        splitstr = lambda s : [w for w in re.split('[^a-zA-Z0-9.]', s) 
                if len(w)>0]
        if isinstance(regionspec,Region):
            return self.key==regionspec.key 
        elif isinstance(regionspec,int):
            # argument is int - a labelindex is expected
            if self.labelindex is None:
                # if this region has no labelindex, see if all children match
                # the given labelindex
                return all([c.might_be(regionspec) for c in self.children])
            else:
                return self.labelindex==regionspec
        elif isinstance(regionspec,str):
            return any([
                    all([w.lower() in splitstr(self.name.lower()) 
                        for w in splitstr(regionspec)]),
                    regionspec==self.key,
                    regionspec==self.name ])
        else:
            raise ValueError("Cannot say if object of type {} might correspond to region".format(type(regionspec)))

    def get_mask(self,space : Space, force=False, resolution=None ):
        """
        Returns a binary mask where nonzero values denote
        voxels corresponding to the region.

        TODO better handling of the case that labelindex is None (use children? stop right away?)

        Parameters
        ----------
        space : Space
            The desired template space.
        force : Boolean (default: False)
            if true, will start large downloads even if they exceed the download
            threshold set in the gbytes_feasible member variable (applies only
            to BigBrain space currently).
        resolution : float or None (Default: None)
            Request the template at a particular physical resolution. If None,
            the native resolution is used.
            Currently, this only works for the BigBrain volume.
        """
        not_avail_msg = 'Parcellation "{}" does not provide a map for space "{}"'.format(
                str(self), str(space) )
        if space not in self.parcellation.maps:
            logger.error(not_avail_msg)
        if len(self.parcellation.maps[space])==0:
            logger.error(not_avail_msg)

        logger.debug("Computing mask for {} in {}".format(
            self.name, space))
        mask = affine = None 

        if self.labelindex is not None:

            maps = self.parcellation.get_maps(space, force=force, resolution=resolution,return_dict=True)
            for description,m in maps.items():
                D = np.array(m.dataobj).squeeze()
                if mask is None: 
                    mask = np.zeros_like(D,dtype=np.uint8)
                    affine = m.affine
                if len(maps)>1 and (description not in self.name):
                    continue
                mask[D==int(self.labelindex)]=1

        if mask is None:
            print("has no own mask:",self," - trying children")
            for child in self.children:
                childmask = child.get_mask(space,force,resolution)
                print("child",child)
                if mask is None:
                    mask = childmask.dataobj
                    affine = childmask.affine
                else:
                    mask = (mask | childmask.dataobj).astype('int')

        if mask is None:
            logger.error("No mask could be computed for the given region "+str(self))
            raise RuntimeError()

        return SpatialImage(dataobj=mask,affine=affine)


    def get_specific_map(self,space,threshold=None):
        """
        Retrieves and returns a specific map of this region, if available
        (otherwise None). This is typically a probability or otherwise
        continuous map, as opposed to the standard label mask from the discrete
        parcellation.

        Parameters
        ----------
        space : Space 
            Template space 
        threshold : float or None
            Threshold for optional conversion to binary mask
        """
        if "maps" not in self.attrs.keys():
            logger.warning("No specific maps known for {}".format(self))
            return None
        if space.id not in self.attrs["maps"].keys():
            logger.warning("No specific map known for {} in space {}.".format(
                self,space))
            return None
        url = self.attrs["maps"][space.id]
        filename = download_file( url )
        if filename is not None:
            img = nib.load(filename)
            if threshold is not None:
                M = (np.asarray(img.dataobj)>threshold).astype('uint8')
                return nib.Nifti1Image(
                        dataobj=M,
                        affine=img.affine,
                        header=img.header)
            else:
                return img
        else:
            logger.warning("Could not retrieve and create Nifti file from {}".format(url))
            return None


    def __str__(self):
        return self.name

    def __repr__(self):
        return  "\n".join("%s%s" % (pre, node.name)
                for pre, _, node in anytree.RenderTree(self))

    def print_tree(self):
        """
        Returns the hierarchy of all descendants of this region as a tree.
        """
        print(self.__repr__())

    def iterate(self):
        """
        Returns an iterator that goes through all regions in this subtree
        (including this parent region)
        """
        return anytree.PreOrderIter(self)


if __name__ == '__main__':

    definition = {
            'name': 'Interposed Nucleus (Cerebellum) - left hemisphere',
            'rgb': [170, 29, 10],
            'labelIndex': 251,
            'ngId': 'jubrain mni152 v18 left',
            'children': [],
            'position': [-9205882, -57128342, -32224599],
            'originDatasets': [ {
                    'kgId': '658a7f71-1b94-4f4a-8f15-726043bbb52a', 
                    'kgSchema': 'minds/core/dataset/v1.0.0', 
                    'filename': 'Interposed Nucleus (Cerebellum) [v6.2, ICBM 2009c Asymmetric, left hemisphere]'
                    }]
            }

