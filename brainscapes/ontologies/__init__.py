def __init_ontology_module__(modulename):
    
    from pkg_resources import resource_listdir,resource_filename
    from os import path
    import json
    import sys

    module = sys.modules[modulename]
    for filename in (resource_listdir(modulename,'')):
        if path.splitext(filename)[-1]=='.json':
            with open(resource_filename(modulename,filename),'r') as f:
                items = json.load(f)
                for item in items:
                    name = item['name'].replace(' ', '_').upper()
                    setattr(module,name,item)

