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
from enum import Enum
from abc import ABC, abstractmethod

class Glossary:
    """
    A very simple class that provides enum-like simple autocompletion for an
    arbitrary list of names.
    """
    def __init__(self,words):
        self.words = list(words)

    def __dir__(self):
        return self.words

    def __str__(self):
        return "\n".join(self.words)

    def __iter__(self):
        return (w for w in self.words)

    def __contains__(self,index):
        return index in self.__dir__()

    def __getattr__(self,name):
        if name in self.words:
            return name
        else:
            raise AttributeError("No such term: {}".format(name))

def create_key(name):
    """
    Creates an uppercase identifier string that includes only alphanumeric
    characters and underscore from a natural language name.
    """
    return re.sub(
            r' +','_',
            "".join([e if e.isalnum() else " " 
                for e in name]).upper().strip() 
            )

class MapType(Enum):
    LABELLED = 1
    CONTINUOUS = 2


class ParcellationIndex:
    """
    Identifies a unique region in a ParcellationMap, combining its labelindex (the "color") and mapindex (the number of the 3Dd map, in case multiple are provided).
    """
    def __init__(self,map,label):
        self.map=map
        self.label=label
    
    def __str__(self):
        return f"({self.map}/{self.label})"

    def __repr__(self):
        return f"{self.__class__.__name__} "+str(self)

    def __eq__(self,other):
        return all([self.map==other.map,self.label==other.label])

    def __hash__(self):
        return hash((self.map,self.label))

