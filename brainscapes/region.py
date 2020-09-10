import json

from brainscapes.ontologies import parcellations,spaces 

class Region:

    """Representation of a region with name and more optional attributes"""

    def __init__(self, definition):
        """
        Constructs a region object from its definition as given in the
        brainscapes parcellation definitions.

        Parameters
        ----------
        definition : dict
        A dictionary of one particular region as formatted in the brainscapes
        parcellation defininition json files.
        """
        self.name = definition['name']
        self.attrs =  definition

    def get_spatial_props(self, space):
        # TODO implement this
        return {}
        # return {
        #     'centroid': '',
        #     'volume': '',
        #     'surface': ''
        # }

    def query_data(self, datatype):
        return None

    def is_cortical(self):
        return True

    def __str__(self):
        return "(name: {0})".format(self.name)

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
