import json
from pkg_resources import resource_filename

class Spaces:

    def __init__(self,filename):
        """
        Instantiates a list or template space definitions from a json file.

        Parameters
        ----------
        filename : string
            Filename of a json file including template space definitions.
            TODO provide a basic schema definition and validator for this
            format.
        """
        with open(filename, 'r') as jsonfile:
            data = json.load(jsonfile)
            for p in data['templateSpaces']:
                short_name = p['shortName'].replace(' ', '_').upper()
                self.__dict__[short_name] = \
                    {
                        'id': p['@id'],
                        'templateUrl': p['templateUrl'],
                        'shortName': short_name
                    }


if __name__ == '__main__':
    print(resource_filename('brainscapes.definitions.atlases','human.json'))
    spaces = Spaces(resource_filename('brainscapes.definitions.atlases','human.json'))
    print(spaces.__dict__)
    print(spaces.BIG_BRAIN)
    print(spaces.ICBM_152_2009C_NONL_ASYM)
    print(spaces.MNI_COLIN_27)
