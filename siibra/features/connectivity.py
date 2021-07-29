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

from .. import logger,parcellations
from ..commons import ParcellationIndex
from ..region import Region
from .feature import RegionalFeature,GlobalFeature
from .query import FeatureQuery
from ..retrieval import cached_gitlab_query,LazyLoader
from collections import defaultdict


GITLAB_SERVER='https://jugit.fz-juelich.de'
GITLAB_PROJECT_ID=3009
QUERY=lambda fname:cached_gitlab_query(
            server=GITLAB_SERVER,project_id=GITLAB_PROJECT_ID,ref_tag="develop",
            folder="connectivity",filename=fname)


class ConnectivityMatrix(GlobalFeature):

    def __init__(self,jsonfile):
        self.src_file = jsonfile
        self._info = json.loads(QUERY(jsonfile))
        self._matrix_loader = LazyLoader(
            func=lambda:self.decode_matrix(jsonfile))
        GlobalFeature.__init__(self,
            self._info['parcellation id'],
            self._info.get('kgId',jsonfile))
        
    @property
    def matrix(self):
        """
        Returns the structured array with region names as row and column field headers.
        """
        return self._matrix_loader.data

    @property
    def array(self):
        """
        Returns the pure data array with connectivity information
        """
        return np.array([self.matrix[f] for f in self.matrix.dtype.names[1:]])

    @property
    def regionnames(self):
        return self._matrix_loader.data.dtype.names[1:]

    @property
    def src_info(self):
        return self._info['description']

    @property
    def src_name(self):
        return self._info['name']

    @property
    def kg_schema(self):
        return self._info['kgschema']

    @property
    def globalrange(self):
        return [fnc(fnc(self._matrix_loader.data[f]) 
                for f in self._matrix_loader.data.dtype.names[1:])
                for fnc in [min,max] ]

    @staticmethod
    def decode_matrix(jsonfile):
        """
        Build a connectivity matrix from json file.
        """
        data = json.loads(QUERY(jsonfile))
        parcellation = parcellations[data['parcellation id']]

        # determine the valid brain regions defined in the file,
        # as well as their indices
        column_names = data['data']['field names']
        valid_regions = {}
        matchings = defaultdict(list)
        for i,name in enumerate(column_names):
            try:
                region = parcellation.decode_region(name,build_group=False)
                if region not in valid_regions.values():
                    valid_regions[i] = region
                    matchings[region].append(name)
                else:
                    logger.debug(f'Region occured multiple times in connectivity dataset: {region}')
            except ValueError:
                continue

        profiles = []
        for i,region in valid_regions.items():
            regionname = column_names[i]
            profile = [region.name]+[data['data']['profiles'][regionname][j] 
                    for j in valid_regions.keys()]
            profiles.append(tuple(profile))
        fields = [('sourceregion','U64')]+[(v.name,'f') for v in valid_regions.values()]
        matrix = np.array(profiles,dtype=fields)
        assert(all(N==len(valid_regions) for N in matrix.shape))
        return matrix

    def __str__(self):
        return "Connectivity matrix for {}".format(self.spec)



class ConnectivityProfile(RegionalFeature):

    show_as_log = True

    def __init__(self, regionspec:str, connectivitymatrix:ConnectivityMatrix, index):
        RegionalFeature.__init__(self,regionspec,connectivitymatrix.dataset_id)
        self._cm_index = index
        self._cm = connectivitymatrix

    @property
    def profile(self):
        return self._cm.matrix[self._cm_index]
    
    @property
    def src_info(self):
        return self._cm.src_info

    @property
    def src_name(self):
        return self._cm.src_name

    @property
    def src_file(self):
        return self._cm.src_file

    @property
    def kg_schema(self):
        return self._cm.kg_schema

    @property
    def regionnames(self):
        return self._cm.regionnames

    @property
    def globalrange(self):
        return self._cm.globalrange

    def __str__(self):
        return f"{self.__class__.__name__} from {self.src_name} for {self.regionspec}"

    def decode(self,parcellation,minstrength=0,force=True):
        """
        Decode the profile into a list of connections strengths to real regions
        that match the given parcellation. 
        If a column name for the profile cannot be decoded, a dummy region is
        returned if force==True, otherwise we fail.
        """
        decoded = ( (strength,parcellation.decode_region(regionname,build_group=not force))
                for strength,regionname in zip(self.profile,self.regionnames)
                if strength>minstrength)
        return sorted(decoded,key=lambda q:q[0],reverse=True)

class ConnectivityProfileQuery(FeatureQuery):

    _FEATURETYPE = ConnectivityProfile

    def __init__(self):
        FeatureQuery.__init__(self)
        jsonfiles = [
            e['name']  for e in json.loads(QUERY(None))
            if e['type']=='blob' and e['name'].endswith('json')]
        for jsonfile in jsonfiles:
            cm = ConnectivityMatrix(jsonfile)
            for parcellation in cm.parcellations:
                for regionname in cm.regionnames:
                    region = parcellation.decode_region(regionname,build_group=False)
                    self.register(ConnectivityProfile(region,cm,regionname))

class ConnectivityMatrixQuery(FeatureQuery):

    _FEATURETYPE = ConnectivityMatrix

    def __init__(self):
        FeatureQuery.__init__(self)
        jsonfiles = [
            e['name']  for e in json.loads(QUERY(None))
            if e['type']=='blob' and e['name'].endswith('.json')]
        for jsonfile in jsonfiles:
            self.register(ConnectivityMatrix(jsonfile))


