import json

from brainscapes.registry import create_key
from brainscapes import parcellations,spaces
import anytree
from brainscapes.retrieval import get_json_from_url
import anytree

def construct_tree(regiondefs,rootname='root',parent=None):
    """ 
    Builds a complete tree from a regions data structure as contained
    inside a brainscapes parcellation ontology. 
    """
    if parent is None:
        root = Region({'name':rootname})
        construct_tree(regiondefs,parent=root)
        return root

    subtrees = []
    for regiondef in regiondefs:
        node = Region(regiondef,parent)
        if "children" in regiondef.keys():
            _ = construct_tree( regiondef['children'],parent=node)
        subtrees.append(node)
    return subtrees


class Region(anytree.NodeMixin):

    """Representation of a region with name and more optional attributes"""

    def __init__(self, definition, parent=None, children=None):
        """
        Constructs a region object from its definition as given in the
        brainscapes parcellation definitions.

        Parameters
        ----------
        definition : dict
            A dictionary of one particular region as formatted in the brainscapes parcellation defininition json files.
        parent : Region
            Parent of this region, if any
        children : list of Region
            Children of this region, if any
        """
        self.name = definition['name']
        self.key = create_key(self.name)
        self.attrs = definition
        if parent is not None:
            self.parent = parent
        if children is not None: 
            self.children = children

    def get_spatial_props(self, space):
        # TODO implement this
        return {}
        # return {
        #     'centroid': '',
        #     'volume': '',
        #     'surface': ''
        # }


    def has_parent(self,parentname):
        return parentname in [a.name for a in self.ancestors]

    def __getattr__(self,name):
        if name in self.attrs.keys():
            return self.attrs[name]
        else:
            raise AttributeError("No such attribute: {}".format(name))

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
        """
        if search_key:
            if exact:
                return anytree.search.find(self, 
                        lambda node: node.key==name)
            else:
                return anytree.search.findall(self,
                        lambda node: name in node.key)
        else:
            if exact:
                return anytree.search.find(self, 
                        lambda node: node.name==name)
            else:
                return anytree.search.findall(self,
                        lambda node: name in node.name)

    def __str__(self):
        """
        Returns the hierarchy of all descendants of this region as a tree.
        """
        return "\n".join("%s%s" % (pre, node.name)
                for pre, _, node in anytree.RenderTree(self))

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

    def __repr__(self):
        return self.name


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
    region = Region(definition)
    spatial_props = region.get_spatial_props(spaces.BIG_BRAIN__HISTOLOGY_)

