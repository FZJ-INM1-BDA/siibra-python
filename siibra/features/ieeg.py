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

from gitlab import Gitlab
import os
import re
import json

from .. import logger,spaces
from .feature import SpatialFeature
from .extractor import FeatureExtractor

DATASETS = {'ca952092-3013-4151-abcc-99a156fe7c83':
    {'server':'https://jugit.fz-juelich.de','project':3009,'folder':'ieeg_contact_points'}
}

class IEEG_Dataset(SpatialFeature):

    def __init__(self,info):
        space = spaces[info['space id']]
        print("space:",space)
        super().__init__(space)
        self.kg_id = info['kgId']
        self.name = info['name']
        self.description = info['description']
        self.electrodes = {} # key: subject_id

    def __str__(self):
        return f"{self.__class__.__name__} {self.kg_id}"

    def new_electrode(self,electrode_id,subject_id):
        return IEEG_Electrode(self,electrode_id,subject_id) # will call register_electrode on construction!

    def register_electrode(self,e):
        if e.subject_id not in self.electrodes:
            self.electrodes[e.subject_id] = {}
        if e.electrode_id in self.electrodes[e.subject_id]:
            logger.warn("Electrode {e.electrode_id} of {e.subject_id} alread registered!")
        self.electrodes[e.subject_id][e.electrode_id] = e
        self._update_location()

    def __iter__(self):
        """
        Iterate over electrodes
        """
        return (self.electrodes[s_id][e_id]
                for s_id in self.electrodes
                for e_id in self.electrodes[s_id])

    def _update_location(self):
        self.location = []
        for e in self:
            if e.location is not None:
                self.location.extend(e.location)

class IEEG_Electrode(SpatialFeature):

    def __init__(self,dataset:IEEG_Dataset,electrode_id,subject_id):
        space = dataset.space
        SpatialFeature.__init__(self,space)
        self.dataset = dataset
        self.electrode_id = electrode_id
        self.subject_id = subject_id
        self.contact_points = {}
        self.n = 0
        dataset.register_electrode(self)

    @property
    def kg_id(self):
        return self.dataset.kg_id

    def __str__(self):
        return f"Electrode {self.electrode_id} of {self.subject_id} with {len(self.contact_points)} contact points ({str(self.dataset)})"

    def new_contact_point(self,id,coord):
        return IEEG_ContactPoint(self,id,coord) # will call register_contact_point on construction!
    
    def register_contact_point(self,contactpoint):
        if contactpoint.id in self.contact_points:
           raise ValueError(f"Contact point with id {contactpoint.id} already registered to {self}") 
        self.contact_points[contactpoint.id] = contactpoint
        self._update_location()

    def __iter__(self):
        """
        Iterate over contact points
        """
        return iter(self.contact_points.values())

    def _update_location(self):
        self.location = [cp.location for cp in self]
        self.dataset._update_location()


class IEEG_ContactPoint(SpatialFeature):
    """
    Basic regional feature for iEEG contact points.
    """
    def __init__(self, electrode, id, coord ):
        SpatialFeature.__init__(self,electrode.space,coord)
        self.electrode = electrode
        self.id = id
        electrode.register_contact_point(self)

    def next(self):
        """
        Returns the next contact point of the same electrode, if any.
        """
        ids_available = list(self.electrode.contact_points.keys())
        my_index = ids_available.index(self.id)
        next_index = my_index+1
        if next_index<len(ids_available):
            next_id = ids_available[next_index]
            return self.electrode.contact_points[next_id]
        else:
            return None

    def prev(self):
        """
        Returns the previous contact point of the same electrode, if any.
        """
        ids_available = list(self.electrode.contact_points.keys())
        my_index = ids_available.index(self.id)
        prev_index = my_index-1
        if prev_index>=0:
            prev_id = ids_available[prev_index]
            return self.electrode.contact_points[prev_id]
        else:
            return None


def load_ptsfile(data):
    sub_id = os.path.basename(data.file_name).split('_')[0]
    result = {'subject_id':sub_id}
    lines = data.decode().decode('utf-8').split("\n")
    N = int(lines[2].strip())
    result['electrodes'] = {}
    for i in range(N):
        fields = re.split('\t',lines[i+3].strip())
        electrode_id,contact_point_id = re.split('(\d+)',fields[0])[:-1]
        if electrode_id not in result['electrodes']:
            result['electrodes'][electrode_id] = {}
        assert(contact_point_id not in result['electrodes'][electrode_id])
        result['electrodes'][electrode_id][contact_point_id] = list(map(float,fields[1:4]))
    return result

def _load_info(server,project,subfolder):
    project = Gitlab(server).projects.get(project)
    f = project.files.get(file_path=os.path.join(subfolder,"info.json"), ref='master')
    return json.loads(f.decode().decode('utf8'))

def _load_files(server,project,subfolder,suffix):
    project = Gitlab(server).projects.get(project)
    files = [f['name'] 
            for f in project.repository_tree(path=subfolder,ref='master',all=True)
            if f['type']=='blob' 
            and f['name'].endswith(suffix)]
    result = []
    for fname in files:
        f = project.files.get(file_path=os.path.join(subfolder,fname), ref='master')
        data = load_ptsfile(f)
        result.append({
            'data':data,
            'fname': fname})
    return result
    
class IEEG_ElectrodeExtractor(FeatureExtractor):

    _FEATURETYPE = IEEG_Electrode
    __pts_files = {}

    def __init__(self,atlas):
        FeatureExtractor.__init__(self,atlas)
        self.load_datasets()
        
    def load_datasets(self):
        """
        Load contact point list and create features.
        """
        for kg_id,spec in DATASETS.items():
            dset = IEEG_Dataset(kg_id,spaces['mni152'])
            if kg_id not in self.__class__.__pts_files:
                self.__class__.__pts_files[kg_id] = _load_files(
                    spec['server'],spec['project'],spec['folder'],'pts')
            for obj in self.__class__.__pts_files[kg_id]: 
                subject_id=obj['data']['subject_id']
                for electrode_id,contact_points in obj['data']['electrodes'].items():
                    e = dset.new_electrode(electrode_id,subject_id)
                    for contact_point_id,coord in contact_points.items():
                        e.new_contact_point(contact_point_id,coord)
                    self.register(e)

class IEEG_ContactPointExtractor(FeatureExtractor):

    _FEATURETYPE = IEEG_ContactPoint
    __pts_files = {}

    def __init__(self,atlas):

        FeatureExtractor.__init__(self,atlas)
        self.load_datasets()
        
    def load_datasets(self):
        """
        Load contact point list and create features.
        """
        for kg_id,spec in DATASETS.items():
            dset = IEEG_Dataset(kg_id,spaces['mni152'])
            if kg_id not in self.__class__.__pts_files:
                self.__class__.__pts_files[kg_id] = _load_files(
                    spec['server'],spec['project'],spec['folder'],'pts')
            for obj in self.__class__.__pts_files[kg_id]: 
                subject_id=obj['data']['subject_id']
                for electrode_id,contact_points in obj['data']['electrodes'].items():
                    e = dset.new_electrode(electrode_id,subject_id)
                    for contact_point_id,coord in contact_points.items():
                        cp = e.new_contact_point(contact_point_id,coord)
                        self.register(cp)


class IEEG_DatasetExtractor(FeatureExtractor):

    _FEATURETYPE = IEEG_Dataset
    __pts_files = {}
    __info = {}


    def __init__(self,atlas):

        FeatureExtractor.__init__(self,atlas)
        self.load_datasets()
        
    def load_datasets(self):
        """
        Load contact point list and create features.
        """
        for kg_id,spec in DATASETS.items():
            if kg_id not in self.__class__.__pts_files:
                self.__class__.__pts_files[kg_id] = _load_files(
                    spec['server'],spec['project'],spec['folder'],'pts')
                self.__class__.__info[kg_id] = _load_info(
                    spec['server'],spec['project'],spec['folder'])
            info = self.__class__.__info[kg_id]
            dset = IEEG_Dataset(info)
            for obj in self.__class__.__pts_files[kg_id]: 
                subject_id=obj['data']['subject_id']
                for electrode_id,contact_points in obj['data']['electrodes'].items():
                    e = dset.new_electrode(electrode_id,subject_id)
                    for contact_point_id,coord in contact_points.items():
                        e.new_contact_point(contact_point_id,coord)
                
            self.register(dset)


if __name__ == '__main__':
    extractor = IEEG_ContactPointExtractor()

