import json
from os import path
from importlib.resources import contents as pkg_contents,path as pkg_path
from . import logger

class Glossary:
    """
    A very simple class that provides enum-like siple autocompletion for an
    arbitrary list of names.
    """
    def __init__(self,words):
        self.words = list(words)

    def __dir__(self):
        return self.words

    def __str__(self):
        return "\n".join(self.words)

    def __iter__(self):
        return (w for w in self.words)

    def __contains__(self,index):
        return index in self.__dir__()

    def __getattr__(self,name):
        if name in self.words:
            return name
        else:
            raise AttributeError("No such term: {}".format(name))

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
        logger.debug("Initializing registry of type {} for {}".format(
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
                        logger.debug("Defining object '{}' with key '{}'".format( obj,key))
                        self.items.append(obj)
                        self.by_key[key] = len(self.items)-1
                        self.by_id[identifier] = len(self.items)-1
                        self.by_name[obj.name] = len(self.items)-1

    def __getitem__(self,index):
        """
        Item access is implemented either by sequential index, key or id.
        """
        if isinstance(index,int) and index<len(self.items):
            return self.items[index]
        elif isinstance(index,self.cls) and (index in self.items):
            # index is itself already an object of this registry - forward
            return index
        elif index in self.by_key:
            return self.items[self.by_key[index]]
        elif index in self.by_id:
            return self.items[self.by_id[index]]
        elif index in self.by_name:
            return self.items[self.by_name[index]]
        else:
            raise ValueError("Cannot access this item in the {} Registry:".format(
                self.cls),index)

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

