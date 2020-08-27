import json
from pathlib import Path


class Spaces:

    def __init__(self):
        path = Path(__file__).parent / '../definitions/atlas_MultiLevelHuman.json'
        with open(path, 'r') as jsonfile:
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
    spaces = Spaces()
    print(spaces.__dict__)
    print(spaces.BIG_BRAIN)
    print(spaces.ICBM_152_2009C_NONL_ASYM)
    print(spaces.MNI_COLIN_27)
