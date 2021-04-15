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

from . import ebrains
from . import logger
from .commons import create_key, Glossary
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
    """

    def __init__(self, name, parcellation, labelindex=None, attrs={}, parent=None, children=None):
        """
        Constructs a new region object.

        Parameters
        ----------
        name : str
            Human-readable name of the rgion
        parcellation : Parcellation
            the parcellation object that this region belongs to
        labelindex : int
            the integer label index used to mark the region in a labelled brain volume
        attrs : dict
            A dictionary of arbitrary additional information
        parent : Region
            Parent of this region, if any
        """
        self.name = name
        self.key = create_key(name)
        # usually set explicitely after constructing an option
        self.parcellation = parcellation
        self.labelindex = labelindex
        self.attrs = attrs
        self.parent = parent
        if children:
            self.children = children
            for c in self.children:
                c.parent = self
                c.parcellation = self.parcellation

    @staticmethod
    def copy(other):
        """
        copy contructor must detach the parent in to avoid problems with
        the Anytree implementation.
        """
        # create an isolated object, detached from the other's tree
        region = Region(other.name, other.parcellation, other.labelindex, other.attrs)

        # Build the new subtree recursively
        region.children = tuple(Region.copy(c) for c in other.children)
        for c in region.children:
            c.parent = region

        return region

    @property
    def labels(self):
        return {r.labelindex for r in self if r.labelindex is not None}

    @property
    def names(self):
        return Glossary([r.key for r in self])

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

    def find(self,regionspec,select_uppermost=False):
        """
        Find region that match the given region specification in the subtree
        headed by this region. 

        Parameters
        ----------
        regionspec : any of 
            - a string with a possibly inexact name, which is matched both
              against the name and the identifier key, 
            - an integer, which is interpreted as a labelindex,
            - a region object
        select_uppermost : Boolean
            If true, only the uppermost matches in the region hierarchy are
            returned (otherwise all siblings as well if they match the name)

        Yield
        -----
        list of matching regions
        """
        result = anytree.search.findall(self,
                lambda node: node.matches(regionspec))
        if len(result)>1 and select_uppermost:
            all_results = result
            mindepth = min([r.depth for r in result])
            result = [r for r in all_results if r.depth==mindepth]
            if len(result)<len(all_results):
                logger.debug("Returning only {} parent nodes of in total {} matching regions for spec '{}'.".format(
                    len(result), len(all_results), regionspec))

        if isinstance(result,Region):
            return [result]
        elif result is None:
            return []
        else:
            return list(result)

    def matches(self,regionspec):
        """ 
        Checks wether this region matches the given region specification. 

        Parameters
        ---------

        regionspec : any of 
            - a string with a possibly inexact name, which is matched both
              against the name and the identifier key, 
            - an integer, which is interpreted as a labelindex,
            - a region object

        Yield
        -----
        True or False
        """
        splitstr = lambda s : [w for w in re.split('[^a-zA-Z0-9.]', s) 
                if len(w)>0]
        if isinstance(regionspec,Region):
            return self.key==regionspec.key 
        elif isinstance(regionspec,int):
            # argument is int - a labelindex is expected
            return self.labelindex==regionspec#(self[regionspec] is not None)
        elif isinstance(regionspec,str):
            # string is given, perform some lazy string matching
            return any([
                    all([w.lower() in splitstr(self.name.lower()) 
                        for w in splitstr(regionspec)]),
                    regionspec==self.key,
                    regionspec==self.name ])
        else:
            raise TypeError(
                    "Cannot interpret region specification of type '{}'".format(
                        type(regionspec)))

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

    def __getitem__(self, labelindex):
        """
        Given an integer label index, return the corresponding region.
        If multiple matches are found, return the unique parent if possible,
        otherwise create an artificial parent node.
        Otherwise, return None

        Parameters
        ----------

        regionlabel: int
            label index of the desired region.

        Return
        ------
        Region object
        """
        if not isinstance(labelindex,int):
            raise TypeError("Index access into the regiontree excepts label indices of integer type")

        # first test this head node
        if self.labelindex==labelindex:
            return self 

        # Consider children, and return the one with smallest depth
        matches = list(filter(lambda x: x is not None,
            [c[labelindex] for c in self.children] ))
        if matches:
            parentmatches = [m for m in matches if m.parent not in matches]
            if len(parentmatches)==1:
                return parentmatches[0]
            else:
                # create an articicial parent region from the multiple matches
                custom_parent = Region._build_grouptree(
                        parentmatches,self.parcellation)
                assert(custom_parent.labelindex==labelindex)
                logger.warn("Label index {} resolves to multiple regions. A customized region subtree is returned: {}".format(
                    labelindex, custom_parent.name))
                return custom_parent
        return None

    @staticmethod
    def _build_grouptree(regions,parcellation):
        """
        Creates an artificial subtree from a list of regions by adding a group
        parent and adding the regions as deep copies recursively.
        """
        # determine appropriate labelindex
        indices = []
        for tree in regions:
            indices.extend([r.labelindex for r in tree])
        unique = set(indices)
        labelindex = next(iter(unique)) if len(unique)==1 else None

        group = Region(
                "Group: "+",".join([r.name for r in regions]),
                parcellation, labelindex, 
                children=[Region.copy(r) for r in regions])
        return group


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

    def __iter__(self):
        """
        Returns an iterator that goes through all regions in this subtree
        (including this parent region)
        """
        return anytree.PreOrderIter(self)

    @staticmethod
    def from_json(jsonstr,parcellation):
        """
        Provides an object hook for the json library to construct a Region
        object from a json definition.
        """

        # first collect any children 
        children = []
        if "children" in jsonstr:
            for regiondef in jsonstr["children"]:
                children.append(Region.from_json(regiondef,parcellation))

        # Then setup the new region object
        assert('name' in jsonstr)
        name = jsonstr['name'] 
        labelindex = jsonstr['labelIndex'] if 'labelIndex' in jsonstr else None
        r = Region(name, parcellation, labelindex, jsonstr, children=children)

        # inherit labelindex from children, if they agree
        if labelindex is None and r.children: 
            L = [c.labelindex for c in r.children]
            if (len(L)>0) and (L.count(L[0])==len(L)):
                r.labelindex = L[0]

        return r

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

