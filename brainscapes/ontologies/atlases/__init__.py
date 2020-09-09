def __init__():
    
    from pkg_resources import resource_listdir,resource_filename
    from os import path
    import json
    from enum import Enum
    import sys

    module = sys.modules[__name__]
    for filename in (resource_listdir('brainscapes.definitions.atlases','')):
        if path.splitext(filename)[-1]=='.json':
            jsonfile = resource_filename('brainscapes.definitions.atlases',filename)
            with open(jsonfile, 'r') as jsonfile:
                data = json.load(jsonfile)
                for parcellation in data['parcellations']:
                    short_name = parcellation['shortName'].replace(' ', '_').upper()
                    setattr(module,short_name,{
                            'id': parcellation['@id'],
                            'availableIn': parcellation['availableIn'],
                            'shortName': short_name
                        })

__init__()
