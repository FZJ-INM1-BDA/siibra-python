import json
from os import path
from importlib.resources import contents as pkg_contents,path as pkg_path
import logging
logging.basicConfig(level=logging.INFO)

class OntologyRegistry:
    """
    A class that registers multiple ontologies from json files by converting
    them to a specific object class based on the object construction function
    provided as constructor parameter.
    """

    def __init__(self,pkgpath,object_hook):
        """
        Populate a new registry from the json files in the package path, using
        the provdided object construction hook function.
        """
        logging.debug("Initializing registry for {}".format(pkgpath))
        self.items = []
        self.by_key = {}
        self.by_id = {}
        for item in pkg_contents(pkgpath):
            if path.splitext(item)[-1]==".json":
                with pkg_path(pkgpath,item) as fname:
                    with open(fname) as f:
                        obj = json.load(f,object_hook=object_hook)
                        key = create_key(str(obj))
                        identifier = obj.id
                        logging.debug("Defining object '{}' with key '{}'".format( obj,key))
                        self.items.append(obj)
                        self.by_key[key] = len(self.items)-1
                        self.by_id[identifier] = len(self.items)-1

    def __getitem__(self,index):
        """
        Item access is implemented either by sequential index, key or id.
        """
        if type(index) is int and index<len(self.items):
            return self.items[index]
        if index in self.by_key:
            return self.items[self.by_key[index]]
        if index in self.by_id:
            return self.items[self.by_id[index]]

    def __dir__(self):
        return list(self.by_key.keys()) + list(self.by_id.keys())

    def __str__(self):
        return "\n".join([i.key for i in self.items])

    def __contains__(self,index):
        return index in self.__dir__()



def create_key(name):
    """
    Creates an uppercase identifier string that includes only alphanumeric
    characters and underscore from a natural language name.
    """
    return "".join(e if e.isalnum() else '_' 
        for e in name.strip()).upper()

