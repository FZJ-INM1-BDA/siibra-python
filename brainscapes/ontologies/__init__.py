
class __OntologyList__:
    """
    A class representing an ontology as given by the brainscapes json
    definitions. Ontology items are directly accessible as object attributes.
    """
    import json

    def __init__(self,filenames):
        """
        Generate an ontology list from json files
        """
        self.attrs = {}
        for fname in filenames:
            with open(fname,'r') as f:
                # NOTE that we assume a list at the top level!
                for item in self.json.load(f):
                    name = item['name'].replace(' ', '_').upper()
                    self.attrs[name] = item

    def __getattr__(self,name):
        if name in self.attrs.keys():
            return self.attrs[name]

    def __dir__(self):
        return self.attrs.keys()


def __init__(): 

    from pkg_resources import resource_filename
    from pkgutil import walk_packages
    from os import path
    from glob import glob

    here = resource_filename(__name__,'')
    for pkg in walk_packages(__path__):
        files = glob(path.join(here,pkg.name,'*.json'))
        globals()[pkg.name.upper()] = __OntologyList__(files)


__init__()
