class Region:
    """Representation of a region with name, referencespace, parcellation and more optional attributes"""
    name = None
    index = None
    referencespace = None
    parcellation = None

    def __init__(self, name, referencespace, parcellation, **kwargs):
        self.name = name
        self.referencespace = referencespace
        self.parcellation = parcellation

    def get_spatial_props(self, space):
        print('Getting spatial props for space: ' + space)
        return {
            'centroid': '',
            'volume': '',
            'surface': ''
        }

    def query_data(self, datatype):
        return None

    def is_cortical(self):
        return True

    def __str__(self):
        return "(name: {0})".format(self.name)

    def __repr__(self):
        return self.__str__()
