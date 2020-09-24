from .registry import Registry,create_key

class Space:

    def __init__(self, identifier, name, template_url=None, ziptarget=None):
        self.id = identifier
        self.name = name
        self.key = create_key(name)
        self.url = template_url
        self.ziptarget = ziptarget

    def __str__(self):
        return self.name

    @staticmethod
    def from_json(obj):
        """
        Provides an object hook for the json library to construct an Atlas
        object from a json stream.
        """
        if '@id' in obj and "minds/core/referencespace/v1.0.0" in obj['@id']:
            if 'templateFile' in obj:
                return Space(obj['@id'], obj['name'], obj['templateUrl'], 
                        ziptarget=obj['templateFile'])
            else:
                return Space(obj['@id'], obj['name'], obj['templateUrl'])
        return obj

REGISTRY = Registry(
        'brainscapes.definitions.spaces', Space )
