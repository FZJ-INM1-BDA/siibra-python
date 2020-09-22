
class Parcellation:

    def __init__(self, identifier, name, version=None):
        self.id = identifier
        self.name = name
        self.version = version
        self.maps = defaultdict(dict)
        self.regions = {}

    def add_map(self, space_id, name, url):
        # TODO check that space_id has a valid object
        self.maps[space_id][name] = url

    def __str__(self):
        return self.name

    @staticmethod
    def from_json(obj):
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

