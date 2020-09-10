from types import ModuleType
import sys

class module(ModuleType):
    """
    Derive from module class in order to add ontology definitions as custom
    module attributes. Automcomplete is provided via a custom __dir__
    implementation.
    NOTE: In some versions of ipython using Jedi for autocompletion, this does
    not work. In an ipython notebook, you might want to use 
    `%config IPCompleter.use_jedi = False` 
    to enable autocompletion of ontology definitions.
    """

    attribute_names = []

    def __init__(self,modulename):
        
        from pkg_resources import resource_listdir,resource_filename
        from os import path
        import json
        import sys

        for filename in (resource_listdir(modulename,'')):
            if path.splitext(filename)[-1]=='.json':
                with open(resource_filename(modulename,filename),'r') as f:
                    items = json.load(f)
                    for item in items:
                        name = item['name'].replace(' ', '_').upper()
                        setattr(self,name,item)
                        self.attribute_names.append(name) 

    def __dir__(self): 
       return self.attribute_names + ModuleType.__dir__(self)


