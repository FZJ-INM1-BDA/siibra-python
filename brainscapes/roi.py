class Roi:
    """
    A representation for a 'region of interest'

    """

    def __init__(self, data, region, hemisphere, threshold):
        self.data = data
        self.region = region
        self.hemisphere = hemisphere
        self.threshold = threshold

    def save(self, filename):
        file = filename if filename.endswith(".nii") else filename+".nii"
        with open(file, 'wb') as f:
            f.write(self.data)

    def __str__(self):
        return "Roi(Region = {0})".format(self.region)
        pass

    def __repr__(self):
        return self.__str__()
