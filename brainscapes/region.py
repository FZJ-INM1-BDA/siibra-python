import json
from pathlib import Path

from brainscapes.parcellations import Parcellations
from brainscapes.spaces import Spaces


class Region:
    """Representation of a region with name and more optional attributes"""
    name = None
    parcellation = None
    index = None

    def __init__(self, name, parcellation, **kwargs):
        self.name = name
        self.parcellation = parcellation

    def get_spatial_props(self, space):
        filename = '../data/regions/' + self.parcellation['shortName'] + '.json'
        path = Path(__file__).parent / filename
        with open(path, 'r') as jsonfile:
            data = json.load(jsonfile)
            for p in data['regions']:
                if p['name'] == self.name:
                    return p
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

    region = Region('Ch 123 (Basal Forebrain) - left hemisphere', Parcellations().CYTOARCHITECTONIC_MAPS)
    spatial_props = region.get_spatial_props(Spaces().BIG_BRAIN['shortName'])
    print(spatial_props)
