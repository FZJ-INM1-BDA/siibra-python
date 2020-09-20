from brainscapes import NAME2IDENTIFIER

class __OntologyList__:
    """
    Represents a set of ontology defintions.
    """
    import json
    import re

    def __init__(self,filenames):
        """
        Generate an ontology list from json files
        """
        self.attrs = {}
        for fname in filenames:
            with open(fname,'r') as f:
                # NOTE that we assume a list at the top level!
                for item in self.json.load(f):
                    name = NAME2IDENTIFIER(item['name'])
                    self.attrs[name] = item
                    # if an @id is available, we also key the same object with its id
                    if '@id' in item.keys():
                        self.attrs[item['@id']] = item

    def __getattr__(self,name):
        if name in self.attrs.keys():
            return self.attrs[name]
        else:
            raise AttributeError("No such attribute: {}".format(
                name) )

    def __getitem__(self, index):
        """
        Access on ontology by its ascii'ed uppercase name (referring to
        brainscapes' NAME2IDENTIFIER func).
        """
        return self.__getattr__(index)


    def __dir__(self):
        return self.attrs.keys()


def __init__(): 

    from glob import glob
    import os

    for path in __path__:
        subfolders = filter(
                os.path.isdir, 
                [os.path.join(path,f) for f in os.listdir(path)])

        for folder in subfolders:
            files = glob(os.path.join(folder,'*.json'))
            name = os.path.basename(folder)
            globals()[name] = __OntologyList__(files)

__init__()
