import json

from brainscapes.ontologies import parcellations,spaces 

class Region:
    """Representation of a region with name and more optional attributes"""
    name = None
    parcellation = None
    index = None

    def __init__(self, name, parcellation, **kwargs):
        self.name = name
        self.parcellation = parcellation

    def get_spatial_props(self, space):
        # TODO implement this
        pass
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

    region = Region(
            'Ch 123 (Basal Forebrain) - left hemisphere', 
            parcellations.CYTOARCHITECTONIC_MAPS)
    spatial_props = region.get_spatial_props(spaces.BIG_BRAIN['shortName'])
    print(spatial_props)
