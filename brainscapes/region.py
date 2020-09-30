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
from brainscapes import parcellations,spaces
from brainscapes.retrieval import get_json_from_url
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

    def find(self,name,exact=True,search_key=False):
        """
        Find region with the given name in all descendants of this region.

        Parameters
        ----------
        name : str
            The name to search for.
        exact : Bool (default: True)
            Wether to return only the exact match (or None if not found), or to
            return a list of all regions whose name contains the given search
            name as a substring (or empty list if none).
        search_key : Bool (default: False)
            If true, the search will compare the region's key instead of name
            (the uppercase variant without special characters)

        Yield
        -----
        list of matching regions
        """
        if search_key:
            if exact:
                result = anytree.search.find(self, 
                        lambda node: node.key==name)
            else:
                result = anytree.search.findall(self,
                        lambda node: name in node.key)
        else:
            if exact:
                result = anytree.search.find(self, 
                        lambda node: node.name==name)
            else:
                result = anytree.search.findall(self,
                        lambda node: name in node.name)
        if isinstance(result,Region):
            return [result]
        elif result is None:
            return []
        else:
            return result

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

    def query_data(self, datatype):
        receptor_data_url = 'https://jugit.fz-juelich.de/t.dickscheid/brainscapes-datafeatures/-/raw/master/receptordata/julichbrain_v1_18.json'
        receptor_data = get_json_from_url(receptor_data_url)
        for data in receptor_data[0]['regions']:
            if data['name'] in self.name:
                return data['files']
        return {}

    # DISABLED, yields a circular import and not need yet
    #def get_receptor_data(self):
        #return receptors.get_receptor_data_by_region(self.name)


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

