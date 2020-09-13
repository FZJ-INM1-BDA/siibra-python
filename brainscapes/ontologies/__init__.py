
class __OntologyList__:
    """
    A class representing an ontology as given by the brainscapes json
    definitions. Ontology items are directly accessible as object attributes.
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
                    name = self.re.sub("[^0-9a-zA-Z]+", "_", item['name']).upper()
                    self.attrs[name] = item

    def __getattr__(self,name):
        if name in self.attrs.keys():
            return self.attrs[name]
        else:
            raise AttributeError("No such attribute: {}".format(
                name) )

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
