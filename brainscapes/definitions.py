import json
from collections import defaultdict
from os import path
from importlib.resources import contents as pkg_contents,path as pkg_path
from brainscapes import NAME2IDENTIFIER
import logging
logging.basicConfig(level=logging.INFO)

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

class AtlasConfiguration:

    def __init__(self, identifier, name):
        self.id = identifier,
        self.name = name
        self.spaces = []
        self.parcellations = []

    def add_space(self, space_id):
        # TODO check that space_id has a valid object
        self.spaces.append(space_id)

    def add_parcellation(self, parcellation_id):
        # TODO check that space_id has a valid object
        self.parcellations.append(parcellation_id)

    def __str__(self):
        return self.name

    @staticmethod
    def from_json(obj):
        if all([ '@id' in obj, 'spaces' in obj, 'parcellations' in obj,
            obj['@id'].startswith("juelich/iav/atlas/v1.0.0") ]):
            p = AtlasConfiguration(obj['@id'], obj['name'])
            for space_id in obj['spaces']:
                p.add_space( space_id )
            for parcellation_id in obj['parcellations']:
                p.add_parcellation( parcellation_id )
            return p
        return obj


keys = {}
ids = {}

spaces = {}
for item in pkg_contents('brainscapes.ontologies.spaces'):
    if path.splitext(item)[-1]==".json":
        with pkg_path('brainscapes.ontologies.spaces',item) as fname:
            with open(fname) as f:
                obj = json.load(f, object_hook=Space.from_json)
                key = NAME2IDENTIFIER(str(obj))
                keys[obj.id]=key
                ids[key]=obj.id
                logging.debug("Defining space '{}' as '{}'".format( obj, key))
                spaces[key] = obj

parcellations = {}
for item in pkg_contents('brainscapes.ontologies.parcellations'):
    if path.splitext(item)[-1]==".json":
        with pkg_path('brainscapes.ontologies.parcellations',item) as fname:
            with open(fname) as f:
                obj = json.load(f, object_hook=Parcellation.from_json)
                key = NAME2IDENTIFIER(str(obj))
                keys[obj.id]=key
                ids[key]=obj.id
                logging.debug("Defining parcellation '{}' as '{}'".format( obj,key))
                parcellations[key] = obj

atlases = {}
for item in pkg_contents('brainscapes.ontologies.atlases'):
    if path.splitext(item)[-1]==".json":
        with pkg_path('brainscapes.ontologies.atlases',item) as fname:
            with open(fname) as f:
                obj = json.load(f, object_hook=AtlasConfiguration.from_json)
                key = NAME2IDENTIFIER(str(obj))
                keys[obj.id]=key
                ids[key]=obj.id
                logging.debug("Defining atlas '{}' as '{}'".format( obj,key))
                atlases[key] = obj

# TODO find a better way to map between @ids and readable uppercase names
id2key = lambda i: keys[i]
key2id = lambda k: ids[k]
