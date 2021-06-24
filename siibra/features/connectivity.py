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

from .. import logger
from ..commons import ParcellationIndex
from ..region import Region
from .feature import RegionalFeature,GlobalFeature
from .extractor import FeatureExtractor
from ..termplot import FontStyles as style
from .. import termplot,parcellations

class ConnectivityProfile(RegionalFeature):

    show_as_log = True

    def __init__(self, region, profile, column_names, src_name, src_info, src_file, parcellation):
        RegionalFeature.__init__(self,region)
        self.profile = profile
        self.src_name = src_name
        self.src_info = src_info
        self.src_file = src_file
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


    def decode(self,parcellation,minstrength=0,force=True):
        """
        Decode the profile into a list of connections strengths to real regions
        that match the given parcellation. 
        If a column name for the profile cannot be decoded, a dummy region is
        returned if force==True, otherwise we fail.
        """
        def decoderegion(parcellation,regionname,force):
            try:
                return parcellation.decode_region(regionname)
            except ValueError as e:
                logger.warning(f'Region name "{regionname}" cannot be decoded by parcellation {parcellation.key}, returning dummy region.')
                if force:
                    return Region(regionname, parcellation, ParcellationIndex(None,None))
                raise e

        decoded = ( (strength,decoderegion(parcellation,regionname,force))
                for strength,regionname in zip(self.profile,self.column_names)
                if strength>minstrength)
        return sorted(decoded,key=lambda q:q[0],reverse=True)

class ConnectivityProfileExtractor(FeatureExtractor):

    _FEATURETYPE = ConnectivityProfile
    __profiles = None

    def __init__(self,**kwargs):

        FeatureExtractor.__init__(self,**kwargs)
        if self.__class__.__profiles is None:
            self.__load_profiles()
        for profile in self.__class__.__profiles :
            self.register(profile)

    def __load_profiles(self):

        project = Gitlab('https://jugit.fz-juelich.de').projects.get(3009)
        jsonfiles = [f['name'] 
                for f in project.repository_tree() 
                if f['type']=='blob' 
                and f['name'].endswith('json')]

        
        minval = maxval = 0
        self.__class__.__profiles  = []
        for jsonfile in jsonfiles: 
            f = project.files.get(file_path=jsonfile, ref='master')
            data = json.loads(f.decode())
            src_name = data['name']
            src_info  = data['description']
            src_file = jsonfile
            parcellation = parcellations[data['parcellation id']]
            if parcellation!=self.parcellation:
                continue
            column_names = data['data']['field names']
            valid_regions = {}
            for i,name in enumerate(column_names):
                try:
                    region = parcellation.decode_region(name)
                except ValueError:
                    continue
                valid_regions[i] = region
            if len(valid_regions)<len(column_names):
                logger.warning(f'{len(valid_regions)} of {len(column_names)} columns in connectivity dataset pointed to valid regions.')
            for i,region in valid_regions.items():
                regionname = column_names[i]
                profile = data['data']['profiles'][regionname]
                if max(profile)>maxval:
                    maxval = max(profile)
                if min(profile)>minval:
                    minval = min(profile)
                self.__class__.__profiles .append( ConnectivityProfile(
                    region, profile, 
                    [r.name for r in valid_regions.values()],
                    src_name, src_info, src_file,
                    parcellation ) )

        for profile in self.__class__.__profiles :
            profile.globalrange = (minval,maxval)


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
    __features = None

    def __init__(self,**kwargs):

        FeatureExtractor.__init__(self,**kwargs)
        if self.__class__.__features is None:
            self._load_features()
        for feature in self.__class__.__features:
            self.register(feature)        

    def _load_features(self):

        project = Gitlab('https://jugit.fz-juelich.de').projects.get(3009)
        jsonfiles = [f['name'] 
                for f in project.repository_tree() 
                if f['type']=='blob' 
                and f['name'].endswith('json')]
        self.__class__.__features = []
        for jsonfile in jsonfiles: 
            f = project.files.get(file_path=jsonfile, ref='master')
            profiles = []
            data = json.loads(f.decode())
            src_name = data['name']
            src_info  = data['description']
            parcellation = parcellations[data['parcellation id']]
            if parcellation!=self.parcellation:
                continue

            # determine the valid brain regions defined in the file,
            # as well as their indices
            column_names = data['data']['field names']
            valid_regions = {}
            for i,name in enumerate(column_names):
                try:
                    region = parcellation.decode_region(name)
                except ValueError:
                    continue
                valid_regions[i] = region
            if len(valid_regions)<len(column_names):
                logger.info(f'{len(valid_regions)} of {len(column_names)} columns in connectivity dataset point to valid regions.')

            for i,region in valid_regions.items():
                regionname = column_names[i]
                profile = [data['data']['profiles'][regionname][i] 
                        for i in valid_regions.keys()]
                profiles.append(profile)
            matrix = np.array(profiles)
            assert(all(N==len(valid_regions) for N in matrix.shape))
            self.__class__.__features.append(
                ConnectivityMatrix(
                    parcellation, matrix, column_names, src_name, src_info ))

if __name__ == '__main__':

    profileextractor = ConnectivityProfileExtractor()
    matrixextractor = ConnectivityMatrixExtractor()

