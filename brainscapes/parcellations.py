from pkg_resources import resource_string, resource_listdir
import json

class Parcellations:
    def __init__(self, filename):
        """
        Instantiates a list or parcellation definitions from a json file.

        Parameters
        ----------
        filename : string
            Filename of a json file including a parcellation definition.
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
        #parcellations = json.load(f)
    #julichbrain = parcellations.CYTOARCHITECTONIC_MAPS
    #print(julichbrain)
    #print(parcellations.CORTICAL_LAYERS_SEGMENTATION)
    #print(parcellations.ISOCORT_NON_ISOCORT_STRUCTURES)
    #print(parcellations.LONG_BUNDLE)
    #print(parcellations.SHORT_BUNDLE)
    #print(parcellations.DIM_64)
    #print(parcellations.DIM_128)
    #print(parcellations.DIM_256)
    #print(parcellations.DIM_512)
    #print(parcellations.DIM_1024)
