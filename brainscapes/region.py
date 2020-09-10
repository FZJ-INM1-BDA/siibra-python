import json
from pathlib import Path

from brainscapes.levels import Levels
from brainscapes.templates import Templates


class Region:
    """Representation of a region with name and more optional attributes"""
    name = None
    level = None
    index = None

    def __init__(self, name, level, **kwargs):
        self.name = name
        self.level = level

    def get_spatial_props(self, template):
        filename = '../data/regions/' + self.level['shortName'] + '.json'
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
    region = Region('Ch 123 (Basal Forebrain) - left hemisphere', Levels().CYTOARCHITECTONIC_MAPS)
    spatial_props = region.get_spatial_props(Templates().BIG_BRAIN['shortName'])
    print(spatial_props)
