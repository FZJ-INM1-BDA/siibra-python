import json
from os import path
from importlib.resources import contents as pkg_contents,path as pkg_path
from brainscapes import NAME2IDENTIFIER, atlas, parcellation, space
import logging
logging.basicConfig(level=logging.INFO)

keys = {}
ids = {}

spaces = {}
for item in pkg_contents('brainscapes.ontologies.spaces'):
    if path.splitext(item)[-1]==".json":
        with pkg_path('brainscapes.ontologies.spaces',item) as fname:
            with open(fname) as f:
                obj = json.load(f, 
                        object_hook=space.Space.from_json)
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
                obj = json.load(f, 
                        object_hook=parcellation.Parcellation.from_json)
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
                obj = json.load(f, 
                        object_hook=atlas.Atlas.from_json)
                key = NAME2IDENTIFIER(str(obj))
                keys[obj.id]=key
                ids[key]=obj.id
                logging.debug("Defining atlas '{}' as '{}'".format( obj,key))
                atlases[key] = obj

# TODO find a better way to map between @ids and readable uppercase names
id2key = lambda i: keys[i]
key2id = lambda k: ids[k]
