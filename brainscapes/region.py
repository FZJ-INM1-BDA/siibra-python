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

from brainscapes.commons import create_key
from brainscapes import parcellations,spaces,ebrains
from brainscapes.retrieval import get_json_from_url
from . import logger
import re
import anytree

def construct_tree(parcellation,entrypoints=None,parent=None):
    """ 
    Builds a complete tree from a regions data structure as contained
    inside a brainscapes parcellation definition. 
    """
    if entrypoints is None:
        root = Region({'name':parcellation.name},parcellation)
        construct_tree(parcellation,parcellation.regions,parent=root)
        return root

    subtrees = []
    for regiondef in entrypoints:
        node = Region(regiondef,parcellation,parent)
        if "children" in regiondef.keys():
            _ = construct_tree( parcellation, regiondef['children'],parent=node)
        subtrees.append(node)
    return subtrees


class Region(anytree.NodeMixin):
    """
    Representation of a region with name and more optional attributes
    TODO implement a Region.from_json factory method to be applied recursively
    from the Parcellation.from_json method
    """

    def __init__(self, definition, parcellation, parent=None, children=None):
        """
        Constructs a region object from its definition as given in the
        brainscapes parcellation configurations.

        Parameters
        ----------
        definition : dict
            A dictionary of one particular region as formatted in the brainscapes parcellation defininition json files.
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
        self.attrs = definition
        if parent is not None:
            self.parent = parent
        if children is not None: 
            self.children = children

    def _ebrains_files(self):
        """
        Returns a list of downloadable files from EBRAINS that could be found
        for this region, if any.
        """
        files = []
        if not all([
            'originDatasets' in self.attrs.keys(),
            len(self.attrs['originDatasets'])>0,
            'kgId' in self.attrs['originDatasets'][0].keys() ]):
            return files
        dataset = self.attrs['originDatasets'][0]['kgId']
        res = ebrains.execute_query_by_id(
                'minds','core','dataset','v1.0.0',
                'brainscapes_files_in_dataset',
                parameters={'dataset':dataset} )
        for dataset in res['results']:
            files.extend(dataset['files'])
        return files

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
            return [result]
        elif result is None:
            return []
        else:
            return result

    def might_be(self,regionspec):
        """ 
        Checks wether this region might match the given specification, which
        could be any of 
            - a string with a name.
            - a region object
        """
        splitstr = lambda s : [w for w in re.split('[^a-zA-Z0-9]', s) if len(w)>0]
        if isinstance(regionspec,Region):
            return self.key==regionspec.key 
        elif isinstance(regionspec,str):
            return any([
                    all([w.lower() in splitstr(self.name.lower()) 
                        for w in splitstr(regionspec)]),
                    regionspec==self.key,
                    regionspec==self.name ])
        else:
            raise ValueError("Cannot say if object of type {} might correspond to region".format(type(regionspec)))

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
    region = Region(definition,parcellations[0])
    spatial_props = region.get_spatial_props(spaces.BIG_BRAIN__HISTOLOGY_)

