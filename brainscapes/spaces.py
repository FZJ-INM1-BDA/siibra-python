import json
from pathlib import Path

# ICBM_152_2009c_NONL_ASYM = 'dafcffc5-4826-4bf1-8ff6-46b8a31ff8e2'
# BIGBRAIN_2015 = 'a1655b99-82f1-420f-a3c2-fe80fd4c8588'
# MNI_Colin_27 = '7f39f7be-445b-47c0-9791-e971c0b6d992'


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
