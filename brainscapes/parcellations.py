import json
from pathlib import Path


class Parcellations:
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
    parcellations = Parcellations()
    jubrain = parcellations.CYTOARCHITECTONIC_MAPS
    print(jubrain)
    print(parcellations.CORTICAL_LAYERS_SEGMENTATION)
    print(parcellations.ISOCORT_NON_ISOCORT_STRUCTURES)
    print(parcellations.LONG_BUNDLE)
    print(parcellations.SHORT_BUNDLE)
    print(parcellations.DIM_64)
    print(parcellations.DIM_128)
    print(parcellations.DIM_256)
    print(parcellations.DIM_512)
    print(parcellations.DIM_1024)
