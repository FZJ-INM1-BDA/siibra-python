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

import os
from enum import Enum

import logging
logger = logging.getLogger(__name__.split(os.path.extsep)[0])
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(name)s:%(levelname)s]  %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class LoggingContext:
    def __init__(self, level ):
        self.level = level
    def __enter__(self):
        self.old_level = logger.level
        logger.setLevel(self.level)
    def __exit__(self, et, ev, tb):
        logger.setLevel(self.old_level)

def set_log_level(level):
    logger.setLevel(level)

set_log_level(os.getenv("SIIBRA_LOG_LEVEL","INFO"))
QUIET = LoggingContext("ERROR")
VERBOSE = LoggingContext("DEBUG")

class Registry:
    """
    Provide attribute-access and iteration to a set of named elements,
    given by a dictionary with keys of 'str' type.
    """
    def __init__(self,matchfunc=lambda a,b:a==b,elements=None):
        """
        Build a glossary from a dictionary with string keys, for easy 
        attribute-like access, name autocompletion, and iteration.
        Matchfunc can be provided to enable inexact matching inside the index operator.
        It is a binary function, taking as first argument a value of the dictionary 
        (ie. an object that you put into this glossary), and as second argument 
        the index/specification that should match one of the objects, and returning a boolean.
        """

        assert(hasattr(matchfunc,'__call__'))
        if elements is None:
            self._elements = {}
        else:
            assert(isinstance(elements,dict))
            assert(all(isinstance(k,str) for k in elements.keys())) 
            self._elements = elements
        self._matchfunc = matchfunc

    def add(self,key,value):
        assert(isinstance(key,str))
        if key in self._elements:
            logger.warn(f"Key {key} already in {__class__.__name__}, existing value will be replaced.")
        self._elements[key] = value

    def __dir__(self):
        return self._elements.keys()

    def __str__(self):
        return f"{self.__class__.__name__}: "+",".join(self._elements.keys())

    def __iter__(self):
        return (w for w in self._elements.keys())

    def __contains__(self,spec):
        return (spec in self._elements)# or any([self._matchfunc(v,spec) for v in self._elements.values()])

    def __len__(self):
        return len(self._elements)

    def __getitem__(self,spec):
        matches = self.find(spec)
        if len(matches)==0:
            print(str(self))
            raise IndexError(f"{__class__.__name__} has no entry matching the specification {spec}")
        elif len(matches)==1:
            return matches[0]
        else:
            S = sorted(matches,reverse=True)
            largest = S[0]
            logger.warning(f"Multiple elements matched the specification '{spec}', and the first in sorting order was chosen: {largest}")
            logger.info(f"Other candidates were: {', '.join(m.name for m in S[1:])}")
            return largest

    def provides(self,spec):
        """
        Returns True if an element that matches the given specification can be found 
        (using find(), thus going beyond the matching of names only as __contains__ does)
        """
        matches = self.find(spec)
        return len(matches)>0

    def find(self,spec):
        """
        Return a list of items matching the given specification, 
        which could be either the name or a specification that 
        works with the matchfunc of the Glossary.
        """
        if isinstance(spec,str) and (spec in self._elements):
            return [self._elements[spec]]
        elif isinstance(spec,int) and (spec<len(self._elements)):
            return [list(self.elements.values())[spec]]
        else:
            return  [v for v in self._elements.values() if self._matchfunc(v,spec)]

    def __getattr__(self,index):
        if index in self._elements:
            return self._elements[index]
        else:
            hint=""
            if isinstance(index,str):
                import difflib
                closest=difflib.get_close_matches(index,list(self._elements.keys()),n=3)
                if len(closest)>0:
                    hint = f"Did you mean {' or '.join(closest)}?"
            raise AttributeError(f"Term '{index}' not in {__class__.__name__}. "+hint)


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


class MapType(Enum):
    LABELLED = 1
    CONTINUOUS = 2
