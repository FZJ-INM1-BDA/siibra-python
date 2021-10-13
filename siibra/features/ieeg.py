# Copyright 2018-2021
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .feature import SpatialFeature
from .query import FeatureQuery

from .. import logger
from ..core.datasets import EbrainsDataset
from ..core.space import Space, Point, PointSet, WholeBrain
from ..retrieval.repositories import GitlabConnector

import re


class IEEG_Dataset(SpatialFeature, EbrainsDataset):
    """
    A Dataset of intracranial EEG measurements retrieved from EBRAINS,
    composed of different sessions on different subjects.
    """
    def __init__(self, dataset_id, name, space):
        SpatialFeature.__init__(self, WholeBrain(space))
        EbrainsDataset.__init__(self, dataset_id, name)
        self.sessions = {}

    def new_session(self, subject_id):
        # NOTE this will call register_session on construction!
        return IEEG_Session(self, subject_id)

    def register_session(self, s):
        if s.subject_id in self.sessions:
            logger.warning(f"Session {str(s)} alread registered!")
        self.sessions[s.subject_id] = s
        self._update_location()

    def __iter__(self):
        """
        Iterate over sessions
        """
        return iter(self.sessions.values())

    def _update_location(self):
        points = []
        for s in self:
            if s.location is not None:
                points.extend(list(s.location))
        if len(points) > 0:
            self.location = PointSet(points, points[0].space)

    @classmethod
    def _from_json(cls, spec):
        return cls(
            dataset_id=spec["kgId"],
            name=spec["name"],
            space=Space.REGISTRY[spec["space id"]],
        )


class IEEG_Session(SpatialFeature):
    """
    An intracranial EEG recording session on a particular subject,
    storing as set of electrodes and linking to an IEEG_Dataset.
    """

    def __init__(self, dataset: IEEG_Dataset, subject_id):
        SpatialFeature.__init__(self, dataset.location)
        self.sub_id = subject_id
        self.dataset = dataset
        self.electrodes = {}  # key: subject_id

    def new_electrode(self, electrode_id):
        return IEEG_Electrode(
            self, electrode_id
        )  # will call register_electrode on construction!

    def register_electrode(self, e):
        if e.electrode_id in self.electrodes:
            logger.warning(
                "Electrode {e.electrode_id} of {e.subject_id} already registered!"
            )
        self.electrodes[e.electrode_id] = e
        self._update_location()

    def __iter__(self):
        """
        Iterate over electrodes
        """
        return iter(self.electrodes.values())

    def _update_location(self):
        points = []
        for electrode in self:
            if electrode.location is not None:
                points.extend(list(electrode.location))
        if len(points) > 0:
            self.location = PointSet(points, points[0].space)
            self.dataset._update_location()


class IEEG_Electrode(SpatialFeature):
    """
    EEG Electrode with multiple contact points placed in a reference space,
    linking to a particular IEEG recording session.
    """
    def __init__(self, session: IEEG_Session, electrode_id):
        SpatialFeature.__init__(self, session.location)
        self.session = session
        self.electrode_id = electrode_id
        self.contact_points = {}
        session.register_electrode(self)

    def new_contact_point(self, id, coord):
        return IEEG_ContactPoint(
            self, id, coord
        )  # will call register_contact_point on construction!

    def register_contact_point(self, contactpoint):
        if contactpoint.id in self.contact_points:
            raise ValueError(
                f"Contact point with id {contactpoint.id} already registered to {self}"
            )
        self.contact_points[contactpoint.id] = contactpoint
        self._update_location()

    def __iter__(self):
        """
        Iterate over contact points
        """
        return iter(self.contact_points.values())

    def _update_location(self):
        points = [cp.location for cp in self if cp.location is not None]
        if len(points) > 0:
            self.location = PointSet(points, self.session.space)
            self.session._update_location()


class IEEG_ContactPoint(SpatialFeature):
    """
    Basic regional feature for iEEG contact points.
    """

    def __init__(self, electrode, id, coord):
        point = Point(coord, electrode.space)
        SpatialFeature.__init__(self, point)
        self.electrode = electrode
        self.id = id
        electrode.register_contact_point(self)

    def next(self):
        """
        Returns the next contact point of the same electrode, if any.
        """
        ids_available = list(self.electrode.contact_points.keys())
        my_index = ids_available.index(self.id)
        next_index = my_index + 1
        if next_index < len(ids_available):
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
        prev_index = my_index - 1
        if prev_index >= 0:
            prev_id = ids_available[prev_index]
            return self.electrode.contact_points[prev_id]
        else:
            return None


def parse_ptsfile(spec):
    lines = spec.split("\n")
    N = int(lines[2].strip())
    result = {}
    for i in range(N):
        fields = re.split("\t", lines[i + 3].strip())
        electrode_id, contact_point_id = re.split(r"(\d+)", fields[0])[:-1]
        if electrode_id not in result:
            result[electrode_id] = {}
        assert contact_point_id not in result[electrode_id]
        result[electrode_id][contact_point_id] = list(map(float, fields[1:4]))
    return result


class IEEG_SessionQuery(FeatureQuery):
    _FEATURETYPE = IEEG_Session
    _CONNECTOR = GitlabConnector("https://jugit.fz-juelich.de", 3009, "master")

    def __init__(self,**kwargs):

        FeatureQuery.__init__(self)
        dset = IEEG_Dataset._from_json(
            self._CONNECTOR.get_loader("ieeg_contact_points/info.json").data
        )

        for fname, loader in self._CONNECTOR.get_loaders("ieeg_contact_points", ".pts"):

            logger.debug(f"Retrieving from {fname}")

            obj = parse_ptsfile(loader.data.decode())
            subject_id = fname.replace('ieeg_contact_points/', '').split("_")[0]
            session = dset.new_session(subject_id)
            for electrode_id, contact_points in obj.items():
                electrode = session.new_electrode(electrode_id)
                for contact_point_id, coord in contact_points.items():
                    electrode.new_contact_point(contact_point_id, coord)
            self.register(session)


if __name__ == "__main__":
    extractor = IEEG_SessionQuery()
