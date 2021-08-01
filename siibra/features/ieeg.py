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

import re
import json

from .. import logger,spaces
from ..retrieval import GitlabLoader
from .feature import SpatialFeature
from .query import FeatureQuery

class IEEG_Dataset(SpatialFeature):

    def __init__(self,dataset_id,name,description,space):
        SpatialFeature.__init__(self,space,dataset_id)
        self.name = name
        self.description = description
        self.sessions = {}

    def new_session(self,subject_id):
        return IEEG_Session(self,subject_id) # will call register_session on construction!

    def register_session(self,s):
        if s.subject_id in self.sessions:
            logger.warn(f"Session {str(s)} alread registered!")
        self.sessions[s.subject_id] = s
        self._update_location()

    def __iter__(self):
        """
        Iterate over sessions
        """
        return iter(self.sessions.values())

    def _update_location(self):
        coords = []
        for s in self:
            if s.location is not None:
                coords.extend(s.location)
        self.location = coords if len(coords)>0 else None

    @staticmethod
    def from_json(spec):
        return IEEG_Dataset(
            dataset_id = spec['kgId'], 
            name = spec['name'], 
            description = spec['description'],
            space = spaces[spec['space id']] )

class IEEG_Session(SpatialFeature):

    def __init__(self,dataset:IEEG_Dataset,subject_id):
        SpatialFeature.__init__(self,dataset.space,dataset.dataset_id)
        self.sub_id = subject_id
        self.dataset = dataset
        self.electrodes = {} # key: subject_id

    def new_electrode(self,electrode_id):
        return IEEG_Electrode(self,electrode_id) # will call register_electrode on construction!

    def register_electrode(self,e):
        if e.electrode_id in self.electrodes:
            logger.warn("Electrode {e.electrode_id} of {e.subject_id} already registered!")
        self.electrodes[e.electrode_id] = e
        self._update_location()

    def __iter__(self):
        """
        Iterate over electrodes
        """
        return iter(self.electrodes.values())

    def _update_location(self):
        coords = []
        for e in self:
            if e.location is not None:
                coords.extend(e.location)
        self.location = coords if len(coords)>0 else None
        self.dataset._update_location()

class IEEG_Electrode(SpatialFeature):

    def __init__(self,session:IEEG_Session,electrode_id):
        SpatialFeature.__init__(self,session.space,session.dataset_id)
        self.session = session
        self.electrode_id = electrode_id
        self.contact_points = {}
        session.register_electrode(self)

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
        coords = [cp.location for cp in self if cp.location is not None]
        self.location = coords if len(coords)>0 else None
        self.session._update_location()


class IEEG_ContactPoint(SpatialFeature):
    """
    Basic regional feature for iEEG contact points.
    """
    def __init__(self, electrode, id, coord ):
        SpatialFeature.__init__(self,electrode.space,electrode.dataset_id,location=coord)
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

def parse_ptsfile(spec):
    lines = spec.split("\n")
    N = int(lines[2].strip())
    result= {}
    for i in range(N):
        fields = re.split('\t',lines[i+3].strip())
        electrode_id,contact_point_id = re.split('(\d+)',fields[0])[:-1]
        if electrode_id not in result:
            result[electrode_id] = {}
        assert(contact_point_id not in result[electrode_id])
        result[electrode_id][contact_point_id] = list(map(float,fields[1:4]))
    return result

class IEEG_SessionQuery(FeatureQuery):
    _FEATURETYPE = IEEG_Session
    _QUERY = GitlabLoader("https://jugit.fz-juelich.de",3009,"master")

    def __init__(self):
        FeatureQuery.__init__(self)
        dset = IEEG_Dataset.from_json(
            json.loads(self._QUERY.get_file('ieeg_contact_points/info.json'))
                )
        for fname,data in self._QUERY.iterate_files('ieeg_contact_points','.pts'):
            obj = parse_ptsfile(data)
            subject_id=fname.split('_')[0]
            session = dset.new_session(subject_id)
            for electrode_id,contact_points in obj.items():
                electrode = session.new_electrode(electrode_id)
                for contact_point_id,coord in contact_points.items():
                    electrode.new_contact_point(contact_point_id,coord)
            self.register(session)

if __name__ == '__main__':
    extractor = IEEG_SessionQuery()

