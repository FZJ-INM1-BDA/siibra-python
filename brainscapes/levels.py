import json
from pathlib import Path


class Levels:
    def __init__(self):
        path = Path(__file__).parent / '../definitions/atlas_MultiLevelHuman.json'
        with open(path, 'r') as jsonfile:
            data = json.load(jsonfile)
            for p in data['parcellations']:
                short_name = p['shortName'].replace(' ', '_').upper()
                self.__dict__[short_name] = \
                    {
                        'id': p['@id'],
                        'availableIn': p['availableIn'],
                        'shortName': short_name
                    }

    def all(self):
        return self.__dict__


if __name__ == '__main__':
    levels = Levels()
    jubrain = levels.CYTOARCHITECTONIC_MAPS
    print(jubrain)
    print(levels.CORTICAL_LAYERS_SEGMENTATION)
    print(levels.ISOCORT_NON_ISOCORT_STRUCTURES)
    print(levels.LONG_BUNDLE)
    print(levels.SHORT_BUNDLE)
    print(levels.DIM_64)
    print(levels.DIM_128)
    print(levels.DIM_256)
    print(levels.DIM_512)
    print(levels.DIM_1024)
