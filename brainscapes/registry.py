import json
from os import path
from importlib.resources import contents as pkg_contents,path as pkg_path
import logging
logging.basicConfig(level=logging.INFO)

class Registry:
    """
    A class that registers semantic definitions from json files by converting
    them to a specific object class based on the object construction function
    provided as constructor parameter.
    """

    def __init__(self,pkgpath,cls):
        """
        Populate a new registry from the json files in the package path, using
        the "from_json" function of the provided class as hook function.
        """
        logging.debug("Initializing registry of type {} for {}".format(
            cls,pkgpath))
        object_hook = cls.from_json
        self.items = []
        self.by_key = {}
        self.by_id = {}
        self.by_name = {}
        self.cls = cls
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
                        self.by_name[obj.name] = len(self.items)-1

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
        if index in self.by_name:
            return self.items[self.by_name[index]]

    def object(self,representation):
        """
        Given one of the used representations, return the corresponding
        object in the registry. Representations are either strings, referring to
        the name, key, or id, or an object pointer (which is then just forwarded).
        """
        if isinstance(representation,str):
            # representation might be key, id, or name
            return self[representation]
        elif  isinstance(representation,self.cls):
            # is already the desired object
            return representation
        else:
            # representation does not represent a known object
            return None

    def __dir__(self):
        return list(self.by_key.keys()) + list(self.by_id.keys())

    def __str__(self):
        return "\n".join([i.key for i in self.items])

    def __contains__(self,index):
        return index in self.__dir__()

    def __getattr__(self,name):
        if name in self.by_key.keys():
            return self.items[self.by_key[name]]
        else:
            raise AttributeError("No such attribute: {}".format(name))


def create_key(name):
    """
    Creates an uppercase identifier string that includes only alphanumeric
    characters and underscore from a natural language name.
    """
    return "".join(e if e.isalnum() else '_' 
        for e in name.strip()).upper()

