# Copyright 2018-2020 Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import numpy as np
import warnings
from gitlab import Gitlab

from .feature import RegionalFeature,GlobalFeature
from .extractor import FeatureExtractor
from ..termplot import FontStyles as style
from .. import retrieval, termplot, parcellations

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
        return "\n".join([style.BOLD+'Connectivity profile of "{}"'.format(self.region)+style.END] 
                + [ termplot.format_row(self.column_names[i],profile[i],calib)
                    for i in np.argsort(profile)
                    if self.profile[i]>0
                    ])

fget_pattern = 'https://jugit.fz-juelich.de/api/v4/projects/3009/repository/files/connectivity%2F{name}/raw?ref=master'
glob_url = 'https://jugit.fz-juelich.de/api/v4/projects/3009/repository/tree?path=connectivity'

class ConnectivityProfileExtractor(FeatureExtractor):

    _FEATURETYPE = ConnectivityProfile
    _SOURCES = {
            '1000brain.json':"https://jugit.fz-juelich.de/api/v4/projects/3009/repository/files/connectivity%2F1000brains.json/raw?ref=master",
            '1000brain_julichbrain_2.5.json':"https://jugit.fz-juelich.de/api/v4/projects/3009/repository/files/connectivity%2F1000brains-v2.5.1.json/raw?ref=master"
            }

    def __init__(self):

        FeatureExtractor.__init__(self)
        for fname,url in ConnectivityProfileExtractor._SOURCES.items():

            minval = maxval = 0
            new_profiles = []
            with open(retrieval.download_file(url,targetname=fname),'r') as f:
                data = json.load(f)
                src_name = data['name']
                src_info  = data['description']
                #parcellation = parcellations[data['parcellation']]
                parcellation = parcellations[data['parcellation id']]
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
        self.src_name = src_name
        self.src_info = src_info

    def __str__(self):
        # TODO implement a reasonable display of the matrix
        return "Connectivity matrix for {}".format(
                self.parcellation)


class ConnectivityMatrixExtractor(FeatureExtractor):

    _FEATURETYPE = ConnectivityMatrix

    def __init__(self):

        FeatureExtractor.__init__(self)

        project = Gitlab('https://jugit.fz-juelich.de').projects.get(3009)
        jsonfiles = [f['name'] 
                for f in project.repository_tree() 
                if f['type']=='blob' 
                and f['name'].endswith('json')]
        for jsonfile in jsonfiles: 
            f = project.files.get(file_path=jsonfile, ref='master')
            profiles = []
            data = json.loads(f.decode())
            src_name = data['name']
            src_info  = data['description']
            parcellation = parcellations[data['parcellation id']]
            column_names = data['data']['field names']
            for region in column_names:
                profiles.append(data['data']['profiles'][region])
            matrix = np.array(profiles)
            assert(all(N==len(column_names) for N in matrix.shape))
            self.register( ConnectivityMatrix(
                parcellation, matrix, column_names, src_name, src_info ))

if __name__ == '__main__':

    profileextractor = ConnectivityProfileExtractor()
    matrixextractor = ConnectivityMatrixExtractor()

