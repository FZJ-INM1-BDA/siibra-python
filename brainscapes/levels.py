from pkg_resources import resource_string, resource_listdir
import json

class Levels:
    def __init__(self, filename):
        """
        Instantiates a list or levels definitions from a json file.

        Parameters
        ----------
        filename : string
            Filename of a json file including a level definition.
            TODO provide a basic schema definition and validator for this
            format.
        """
        with open(filename, 'r') as jsonfile:
            data = json.load(jsonfile)
            for parcellation in data['parcellations']:
                short_name = parcellation['shortName'].replace(' ', '_').upper()
                self.__dict__[short_name] = {
                        'id': parcellation['@id'],
                        'availableIn': parcellation['availableIn'],
                        'shortName': short_name
                    }

    def all(self):
        return self.__dict__


if __name__ == '__main__':

    # We include the json definitions as non-python package-data into the library.
    # see https://stackoverflow.com/questions/1612733/including-non-python-files-with-setup-py
    print(resource_listdir('brainscapes.definitions.atlases',''))
    #with importlib.resources.path("brainscapes.definitions", "atlas_MultiLevelHuman.json") as f:
        #levels = json.load(f)
    #julichbrain = levels.CYTOARCHITECTONIC_MAPS
    #print(julichbrain)
    #print(levels.CORTICAL_LAYERS_SEGMENTATION)
    #print(levels.ISOCORT_NON_ISOCORT_STRUCTURES)
    #print(levels.LONG_BUNDLE)
    #print(levels.SHORT_BUNDLE)
    #print(levels.DIM_64)
    #print(levels.DIM_128)
    #print(levels.DIM_256)
    #print(levels.DIM_512)
    #print(levels.DIM_1024)
