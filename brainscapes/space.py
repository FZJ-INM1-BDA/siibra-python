
class Space:

    def __init__(self, identifier, name, template_url=None):
        self.id = identifier
        self.name = name
        self.template_url = template_url

    def __str__(self):
        return self.name

    @staticmethod
    def from_json(obj):
        if '@id' in obj and "minds/core/referencespace/v1.0.0" in obj['@id']:
            return Space(obj['@id'], obj['name'], obj['templateUrl'])
        return obj

