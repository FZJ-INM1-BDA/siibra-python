class Region:
    """Representation of a region with name, referencespace, parcellation and more optional attributes"""
    name = None
    referencespace = None
    parcellation = None
    _name = None
    _area = None
    _average_orientation = None
    _centres = None
    _cortical = None
    _tract_lengths = None
    _volume = None
    _weights = None
    _rgb = None
    _position = None
    _status = None
    _links = None

    def __init__(self, name, referencespace, parcellation, **kwargs):
        self.name = name
        self.referencespace = referencespace
        self.parcellation = parcellation
        self._area = kwargs.get('area', None)
        self._average_orientation = kwargs.get('averageOrientation', None)
        self._centres = kwargs.get('centres', None)
        self._cortical = kwargs.get('cortical', None)
        self._tract_lengths = kwargs.get('tractLengths', None)
        self._volume = kwargs.get('volume', None)
        self._weights = kwargs.get('weights', None)
        self._rgb = kwargs.get('rgb', None)
        self._position = kwargs.get('position', None)
        self._status = kwargs.get('status', None)
        self._links = kwargs.get('links', None)

    def get_spatial_props(self, space):
        print('Getting spatial props for space: ' + space)
        return None

    def __str__(self):
        return "(name: {0}, rgb: {1})".format(self.name, self._rgb)

    def __repr__(self):
        return self.__str__()
