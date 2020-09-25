import json
import numpy as np
import warnings
from brainscapes import retrieval
from brainscapes.features.feature import RegionalFeature,GlobalFeature,FeaturePool
from brainscapes import parcellations
from brainscapes import termplot 

class ConnectivityProfile(RegionalFeature):

    show_as_log = True

    def __init__(self, region, profile, column_names, src_name, src_info, parcellation):
        RegionalFeature.__init__(self,region)
        self.profile = profile
        self.src_name = src_name
        self.src_info = src_info
        self.column_names = column_names
        self.parcellation = parcellation
        self.globalrange = None

    def __str__(self):
        """
        Returns a multiline barplot of the sorted profiles in log scale.
        Set ConnectivityProfile.show_as_log to False to see original values
        """
        if ConnectivityProfile.show_as_log:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                profile = np.log(self.profile)
                vrange = np.log(self.globalrange)
            profile[profile==-np.inf] = 0
            vrange[vrange==-np.inf] = 0
        else:
            profile = self.profile
            vrange = self.globalrange
        calib = termplot.calibrate(vrange, self.column_names)
        return "\n".join([
            termplot.format_row(self.column_names[i],profile[i],calib)
            for i in np.argsort(profile)
            if self.profile[i]>0
            ])

class ConnectivityProfileParser(FeaturePool):

    _FEATURETYPE = ConnectivityProfile
    _SOURCES = {
            '1000brain.json':"https://jugit.fz-juelich.de/api/v4/projects/3009/repository/files/connectivity%2F1000brains.json/raw?ref=master"
            }

    def __init__(self):

        FeaturePool.__init__(self)
        for fname,url in ConnectivityProfileParser._SOURCES.items():

            minval = maxval = 0
            new_profiles = []
            with open(retrieval.download_file(url,targetname=fname),'r') as f:
                data = json.load(f)
                src_name = data['name']
                src_info  = data['description']
                parcellation = parcellations[data['parcellation']]
                regions_available = data['data']['field names']
                for region in regions_available:
                    profile = data['data']['profiles'][region]
                    if max(profile)>maxval:
                        maxval = max(profile)
                    if min(profile)>minval:
                        minval = min(profile)
                    new_profiles.append( ConnectivityProfile(
                        region, profile, 
                        regions_available,
                        src_name, src_info, 
                        parcellation ) )

        for profile in new_profiles:
            profile.globalrange = (minval,maxval)
            self.register(profile)


class ConnectivityMatrix(GlobalFeature):

    def __init__(self, parcellation, matrix, column_names, src_name, src_info ):
        GlobalFeature.__init__(self,parcellation)
        self.matrix = matrix
        self.column_names = column_names

    def __str__(self):
        # TODO implement a reasonable display of the matrix
        return "Connectivity matrix for {}".format(
                self.parcellation)


class ConnectivityMatrixParser(FeaturePool):

    _FEATURETYPE = ConnectivityMatrix
    _SOURCES = {
            '1000brain.json':"https://jugit.fz-juelich.de/api/v4/projects/3009/repository/files/connectivity%2F1000brains.json/raw?ref=master"
            }

    def __init__(self):

        FeaturePool.__init__(self)
        for fname,url in ConnectivityMatrixParser._SOURCES.items():

            profiles = []
            with open(retrieval.download_file(url,targetname=fname),'r') as f:
                data = json.load(f)
                src_name = data['name']
                src_info  = data['description']
                parcellation = parcellations[data['parcellation']]
                column_names = data['data']['field names']
                for region in column_names:
                    profiles.append(data['data']['profiles'][region])

            matrix = np.array(profiles)
            assert(all(N==len(column_names) for N in matrix.shape))
            self.register( ConnectivityMatrix(
                parcellation, matrix, column_names, src_name, src_info ))

if __name__ == '__main__':

    profilepool = ConnectivityProfileParser()
    matrixpool = ConnectivityMatrixParser()

