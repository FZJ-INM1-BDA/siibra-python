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
