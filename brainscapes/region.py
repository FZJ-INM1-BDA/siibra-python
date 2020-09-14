import json

from brainscapes.ontologies import parcellations,spaces 
from anytree import NodeMixin

class Region(NodeMixin):

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

    def query_data(self, datatype):
        return None

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


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
    spatial_props = region.get_spatial_props(spaces.BIG_BRAIN['shortName'])
    print(spatial_props)
