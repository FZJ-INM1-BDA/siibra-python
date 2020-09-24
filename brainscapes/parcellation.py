from .registry import Registry,create_key
from collections import defaultdict

class Parcellation:

    def __init__(self, identifier, name, version=None):
        self.id = identifier
        self.name = name
        self.key = create_key(name)
        self.version = version
        self.maps = defaultdict(dict)
        self.regions = {}

    def add_map(self, space_id, name, url):
        # TODO check that space_id has a valid object
        self.maps[space_id][name] = url

    def __str__(self):
        return self.name

    def __eq__(self,other):
        """
        Compare this parcellation with other objects. If other is a string,
        compare to key, name or id.
        """
        if isinstance(other,Parcellation):
            return self.id==other.id
        elif isinstance(other,str):
            return any([
                self.name==other, 
                self.key==other,
                self.id==other])
        else:
            raise ValueError("Cannot compare object of type {} to Parcellation".format(type(other)))

    @staticmethod
    def from_json(obj):
        """
        Provides an object hook for the json library to construct an Atlas
        object from a json stream.
        """
        if '@id' in obj and 'maps' in obj:
            if 'version' in obj:
                p = Parcellation(obj['@id'], obj['name'], obj['version'])
            else:
                p = Parcellation(obj['@id'], obj['name'])
            for space_id,maps in obj['maps'].items():
                for name, url in maps.items():
                    p.add_map( space_id, name, url) 
            # TODO model the regions already here as a hierarchy tree
            if 'regions' in obj:
                p.regions = obj['regions']
            return p
        return obj

REGISTRY = Registry(
        'brainscapes.definitions.parcellations', Parcellation.from_json )
