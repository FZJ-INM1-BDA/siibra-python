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

import io
import PIL.Image as Image
from os import path
from collections import namedtuple
import re

from .feature import RegionalFeature
from .extractor import FeatureExtractor
from ..authentication import Authentication
from ..termplot import FontStyles as style
from .. import ebrains, retrieval, logger

try:
    import matplotlib.pyplot as plt
    HAVE_PLT=True
except Exception as e:
    HAVE_PLT=False


RECEPTOR_SYMBOLS = {
        "5-HT1A": { 
            "receptor" : {
                "latex" : "$5-HT_{1A}$", 
                "markdown" : "5-HT<sub>1A</sub>", 
                "name" : "5-hydroxytryptamine receptor type 1A" 
                }, 
            'neurotransmitter' : { 
                "label" : "5-HT",
                "latex" : "$5-HT$",
                "markdown" : "5-HT",
                "name" : "serotonin",
                } 
            },
        "5-HT2": {
            "receptor" : {
                "latex" : "$5-HT_2$",
                "markdown" : "5-HT<sub>2</sub>",
                "name" : "5-hydroxytryptamine receptor type 2" 
                },
            'neurotransmitter' : {
                "label" : "5-HT",
                "latex" : "$5-HT$",
                "markdown" : "5-HT",
                "name" : "serotonin" 
                } 
            } ,
        "AMPA": { 
            "receptor" : {
                "latex" : "$AMPA$",
                "markdown" : "AMPA",
                "name" : "alpha-amino-3hydroxy-5-methyl-4isoxazolepropionic acid receptor" 
                },
            'neurotransmitter' : { 
                "label" : "Glu",
                "latex" : "$Glu$",
                "markdown" : "Glu",
                "name" : "glutamate" 
                } 
            },
        "BZ": { 
            "receptor" : {
                "latex" : "$GABA_A/BZ$",
                "markdown" : "GABA<sub>A</sub>/BZ",
                "name" : "gamma-aminobutyric acid receptor type A / benzodiazepine associated binding site" 
                },
            'neurotransmitter' : { 
                "label" : "GABA",
                "latex" : "$GABA$",
                "markdown" : "GABA",
                "name" : "gamma-aminobutyric acid" 
                } 
            },
        "D1": { 
            "receptor" : {
                "latex" : "$D_1$", 
                "markdown" : "D<sub>1</sub>", 
                "name" : "dopamine receptor type 1" 
                } , 
            'neurotransmitter' : { 
                "label" : "DA", 
                "latex" : "$DA$", 
                "markdown" : "DA", 
                "name" : "dopamine" 
                }
            },
            "GABAA": {
                    "receptor" : {
                        "latex" : "$GABA_A$",
                        "markdown" : "GABA<sub>A</sub>",
                        "name" : "gamma-aminobutyric acid receptor type A" 
                        },
                    'neurotransmitter' : {
                        "label" : "GABA",
                        "latex" : "$GABA$",
                        "markdown" : "GABA",
                        "name" : "gamma-aminobutyric acid" 
                        }
                    },
        "GABAB": {
                "receptor" : {
                    "latex" : "$GABA_B$",
                    "markdown" : "GABA<sub>B</sub>",
                    "name" : "gamma-aminobutyric acid receptor type B" 
                    },
                'neurotransmitter' : {
                    "label" : "GABA",
                    "latex" : "$GABA$",
                    "markdown" : "GABA",
                    "name" : "gamma-aminobutyric acid" }
                },
        "M1": {
                "receptor" : {
                    "latex" : "$M_1$",
                    "markdown" : "M<sub>1</sub>",
                    "name" : "muscarinic acetylcholine receptor type 1" 
                    },
                'neurotransmitter' : {
                    "label" : "ACh",
                    "latex" : "$ACh$",
                    "markdown" : "ACh",
                    "name" : "acetylcholine" 
                    }
                },
        "M2": {
                "receptor" : {
                    "latex" : "$M_2$",
                    "markdown" : "M<sub>2</sub>",
                    "name" : "muscarinic acetylcholine receptor type 2" 
                    },
                'neurotransmitter' : {
                    "label" : "ACh",
                    "latex" : "$ACh$",
                    "markdown" : "ACh",
                    "name" : "acetylcholine" 
                    }
                },
        "M3": {
                "receptor" : {
                    "latex" : "$M_3$",
                    "markdown" : "M<sub>3</sub>",
                    "name" : "muscarinic acetylcholine receptor type 3" 
                    },
                'neurotransmitter' : {
                    "label" : "ACh",
                    "latex" : "$ACh$",
                    "markdown" : "ACh",
                    "name" : "acetylcholine" 
                    }
                },
        "NMDA": {
                "receptor" : {
                    "latex" : "$NMDA$",
                    "markdown" : "NMDA",
                    "name" : "N-methyl-D-aspartate receptor" 
                    },
                'neurotransmitter' : {
                    "label" : "Glu",
                    "latex" : "$Glu$",
                    "markdown" : "Glu",
                    "name" : "glutamate" 
                    }
                },
        "alpha1": {
                "receptor" : {
                    "latex" : "$\\alpha_1$",
                    "markdown" : "&#945<sub>1</sub>",
                    "name" : "alpha-1 adrenergic receptor" 
                    },
                'neurotransmitter' : {
                    "label" : "NE",
                    "latex" : "$NE$",
                    "markdown" : "NE",
                    "name" : "norepinephrine" 
                    }
                },
        "alpha2": {
                "receptor" : {
                    "latex" : "$\\alpha_2$",
                    "markdown" : "&#945<sub>2</sub>",
                    "name" : "alpha-2 adrenergic receptor" 
                    },
                'neurotransmitter' : {
                    "label" : "NE",
                    "latex" : "$NE$",
                    "markdown" : "NE",
                    "name" : "norepinephrine" 
                    }
                },
        "alpha4beta2": {
                "receptor" : {
                    "latex" : "$\\alpha_4\\beta_2$",
                    "markdown" : "&#945<sub>4</sub>&#946<sub>2</sub>",
                    "name" : "alpha-4 beta-2 nicotinic receptor" 
                    },
                'neurotransmitter' : {
                    "label" : "ACh",
                    "latex" : "$ACh$",
                    "markdown" : "ACh",
                    "name" : "acetylcholine"
                    }
                },
        "kainate": {
                "receptor" : {
                    "latex" : "$kainate$",
                    "markdown" : "kainate",
                    "name" : "kainate receptors" 
                    },
                'neurotransmitter' : {
                    "label" : "Glu",
                    "latex" : "$Glu$",
                    "markdown" : "Glu",
                    "name" : "glutamate"
                    }
                },
        "mGluR2_3": {
                "receptor" : {
                    "latex" : "$mGluR2/3$",
                    "markdown" : "mGluR2/3",
                    "name" : "metabotropic glutamate receptor type 2 and 3"
                    },
                'neurotransmitter' : {
                    "label" : "Glu",
                    "latex" : "$Glu$",
                    "markdown" : "Glu",
                    "name" : "glutamate"
                    }
                }
        }

def get_bytestream_from_file(url):
    fname = retrieval.download_file(url)
    with open(fname, 'rb') as f:
        return io.BytesIO(f.read())

def unify_stringlist(L: list):
    """ Adds asterisks to strings that appear multiple times, so the resulting
    list has only unique strings but still the same length, order, and meaning. 
    For example: 
        unify_stringlist(['a','a','b','a','c']) -> ['a','a*','b','a**','c']
    """
    assert(all([isinstance(l,str) for l in L]))
    return [L[i]+"*"*L[:i].count(L[i]) for i in range(len(L))]

def edits1(word):
    """
    Produces a list of  all possible edits of a given word that can be produced
    by applying a single character modification.
    From Peter Norvig, see http://norvig.com/spell-correct.html.
    """
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def decode_tsv(url):
    bytestream = get_bytestream_from_file(url)
    header = bytestream.readline()
    lines = [l.strip() 
            for l in bytestream.readlines() 
            if len(l.strip())>0]
    sep = b'{' if b'{' in lines[0] else b'\t' 
    keys = unify_stringlist([
        n.decode('utf8').replace('"','').replace("'","").strip() 
        for n in header.split(sep)])
    if any(k.endswith('*') for k in keys):
        logger.debug('Redundant headers: {} in file {}'.format(
            "/".join(keys), url))
    assert(len(keys)==len(set(keys)))
    return  { l.split(sep)[0].decode('utf8') : dict(
        zip(keys, [re.sub(r"\\+",r"\\",v.decode('utf8').strip()) for v in l.split(sep)])) 
        for l in lines }


Density = namedtuple('Density','name, mean, std, unit')

class DensityFingerprint():

    unit = None
    labels = []
    meanvals = []
    stdvals = []
    n = 0

    def __init__(self,datadict):
        """
        Create a DensityFingerprint from a data dictionary coming from a
        receptor fingerprint tsv file.
        """
        units = {list(v.values())[3] for v in datadict.values()}
        assert(len(units)==1)
        self.unit=next(iter(units))
        self.labels=list(datadict.keys())
        try:
            mean=[datadict[l]['density (mean)'] for l in self.labels]
            std=[datadict[l]['density (sd)'] for l in self.labels]
        except KeyError as e: 
            print(str(e))
            logger.error('Could not parse fingerprint from this dictionary')
        self.meanvals=[float(m) if m.isnumeric() else 0 for m in mean]
        self.stdvals=[float(s) if s.isnumeric() else 0 for s in std]

    def __getitem__(self,index):
        assert(index<len(self.labels))
        return Density(
                name=self.labels[index],
                mean=self.meanvals[index],
                std=self.stdvals[index],
                unit=self.unit)

    def __iter__(self):
            self.n = 0
            return self

    def __next__(self):
        if self.n < len(self.labels):
            self.n += 1
            return self[self.n-1]
        else:
            raise StopIteration

    def __str__(self):
        return "\n".join(
                "{d.name:15.15s} {d.mean:8.1f} {d.unit} (+/-{d.std:5.1f})".format(d=d)
                for d in iter(self))


class ReceptorDistribution(RegionalFeature):
    """
    Reprecent a receptor distribution dataset with fingerprint, profiles and
    autoradiograph samples. This implements a lazy loading scheme.
    TODO lazy loading could be more elegant.
    """

    def __init__(self, region, kg_result):

        RegionalFeature.__init__(self,region)
        self.active = False
        self.name = kg_result["name"]
        self.files = kg_result["files"]
        self.info = kg_result['description']
        self.identifier = kg_result['identifier']
        self.url = "https://search.kg.ebrains.eu/instances/Dataset/{}".format(self.identifier)
        self.modality = kg_result['modality']
        self.__profiles = {}
        self.__autoradiographs = {}
        self.__fingerprint = None
        self.__profile_unit = None

    @property
    def autoradiographs(self):
        self._load()
        return self.__autoradiographs

    @property
    def fingerprint(self):
        self._load()
        return self.__fingerprint

    @property
    def profile_unit(self):
        self._load()
        return self.__profile_unit

    @property
    def profiles(self):
        self._load()
        return self.__profiles

    def _load(self):

        if self.active:
            return

        logger.debug('Loading receptor data for'+self.region)

        for url in self.files:

            # Receive cortical profiles, if any
            if '_pr_' in url:
                suffix = path.splitext(url)[-1]
                if suffix == '.tsv':
                    rtype, basename = url.split("/")[-2:]
                    if rtype in basename:
                        data = decode_tsv(url)
                        units = {list(v.values())[3] for v in data.values()}
                        assert(len(units)==1)
                        self.__profile_unit=next(iter(units))
                        # column headers are sometimes messed up, so we fix the 2nd value
                        densities = {int(k):float(list(v.values())[2]) 
                                for k,v in data.items()
                                if k.isnumeric()}
                        rtype = self._check_rtype(rtype)
                        self.__profiles[rtype] = densities

            # Receive autoradiographs, if any
            if '_ar_' in url:
                rtype, basename = url.split("/")[-2:]
                if rtype in basename:
                    rtype = self._check_rtype(rtype)
                    self.__autoradiographs[rtype] = url

            # receive fingerprint, if any
            if '_fp_' in url:
                data = decode_tsv(url) 
                self.__fingerprint = DensityFingerprint(data) 

        self.active = True

    def _check_rtype(self,rtype):
        """ 
        Verify that the receptor type name matches the symbol table. 
        Return if ok, fix if a close match is found, raise Excpetion otherwise.
        """
        if rtype in RECEPTOR_SYMBOLS.keys():
            return rtype
        # fix if only different by 1 letter from a close match in the symbol table
        close_matches = list(edits1(rtype).intersection(RECEPTOR_SYMBOLS.keys()))
        if len(close_matches)==1:
            prev,new = rtype, close_matches[0]
            logger.debug("Receptor type identifier '{}' replaced by '{}' for {}".format(
                prev,new, self.region))
            return new
        else:
            logger.warning("Receptor type identifier '{}' is not listed in the corresponding symbol table of region {}. Please verify.".format(
                        rtype, self.region))
            return rtype

    def __repr__(self):
        self._load()
        return self.__str__()

    def __str__(self):
        """ Outputs a small table of available profiles and autoradiographs. """
        self._load()
        #if len(self.profiles)+len(self.autoradiographs)==0:
                #return style.BOLD+"Receptor density for area {}".format(self.region)+style.END
        return "\n"+"\n".join(
                [style.BOLD+"Receptor densities for area {}".format(self.region)+style.END] +
                [style.ITALIC+"{!s:20} {!s:>8} {!s:>15} {!s:>11}".format(
                    'Type','profile','autoradiograph','fingerprint')+style.END] +
                ["{!s:20} {!s:>8} {!s:>15} {!s:>11}".format(
                    rtype,
                    'x'*(rtype in self.profiles),
                    'x'*(rtype in self.autoradiographs),
                    'x'*(rtype in self.fingerprint))
                    for rtype in RECEPTOR_SYMBOLS.keys()
                    if (rtype in self.profiles 
                        or rtype in self.autoradiographs)] )

    def plot(self,title=None):
        if not HAVE_PLT:
            logger.warning('matplotlib.pyplot not available to siibra. plotting disabled.')
            return None

        self._load()
        import numpy as np
        from collections import deque

        # plot profiles and fingerprint
        fig = plt.figure(figsize=(8,3))
        plt.subplot(121)
        for _,profile in self.profiles.items():
            plt.plot(list(profile.keys()),np.fromiter(profile.values(),dtype='d'))
        plt.xlabel('Cortical depth (%)')
        plt.ylabel("Receptor density\n({})".format(self.profile_unit))
        plt.grid(True)
        if title is not None:
            plt.title(title)
        plt.legend(labels=[l for l in self.profiles],
                   loc="center right", prop={'size': 5})

        ax = plt.subplot(122,projection='polar')
        angles = deque(np.linspace(0, 2*np.pi,  len(self.fingerprint.labels)+1)[:-1][::-1])
        angles.rotate(5)
        angles = list(angles)
        means = [d.mean for d in self.fingerprint]
        stds = [d.mean+d.std for d in self.fingerprint]
        plt.plot(angles+[angles[0]],means+[means[0]],'k-',lw=3)
        plt.plot(angles+[angles[0]],stds+[stds[0]],'k',lw=1)
        ax.set_xticks(angles)
        ax.set_xticklabels([l for l in self.fingerprint.labels])
        #ax.set_yticklabels([])
        ax.tick_params(pad=9,labelsize=9)
        ax.tick_params(axis='y',labelsize=6)
        return fig


class ReceptorQuery(FeatureExtractor):

    _FEATURETYPE = ReceptorDistribution

    def __init__(self):

        FeatureExtractor.__init__(self)
        kg_query = ebrains.execute_query_by_id('minds', 'core', 'dataset', 'v1.0.0', 'siibra_receptor_densities')
        for kg_result in kg_query['results']:
            region_names = [e['name'] for e in kg_result['region']]
            for region_name in region_names:
                feature = ReceptorDistribution(region_name,kg_result)
                self.register(feature)


if __name__ == '__main__':

    auth = Authentication.instance()
    auth.set_token('eyJhbGci....')
    extractor = ReceptorQuery()
    print(extractor)
