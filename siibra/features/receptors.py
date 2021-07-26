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

import PIL.Image as Image
import numpy as np
from io import BytesIO
from os import path
from collections import namedtuple
import re

from .feature import RegionalFeature
from .query import FeatureQuery
from ..ebrains import Authentication
from ..termplot import FontStyles as style
from .. import ebrains, logger
from ..retrieval import LazyLoader

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

def unify_stringlist(L: list):
    """ Adds asterisks to strings that appear multiple times, so the resulting
    list has only unique strings but still the same length, order, and meaning. 
    For example: 
        unify_stringlist(['a','a','b','a','c']) -> ['a','a*','b','a**','c']
    """
    assert(all([isinstance(l,str) for l in L]))
    return [L[i]+"*"*L[:i].count(L[i]) for i in range(len(L))]

def decode_tsv(bytearray):
    bytestream = BytesIO(bytearray)
    header = bytestream.readline()
    lines = [l.strip() 
            for l in bytestream.readlines() 
            if len(l.strip())>0]
    sep = b'{' if b'{' in lines[0] else b'\t' 
    keys = unify_stringlist([
        n.decode('utf8').replace('"','').replace("'","").strip() 
        for n in header.split(sep)])
    if any(k.endswith('*') for k in keys):
        logger.warn('Redundant headers in receptor file')
    assert(len(keys)==len(set(keys)))
    return  { l.split(sep)[0].decode('utf8') : dict(
        zip(keys, [re.sub(r"\\+",r"\\",v.decode('utf8').strip()) for v in l.split(sep)])) 
        for l in lines }



class DensityProfile():

    def __init__(self,data):
        units = {list(v.values())[3] for v in data.values()}
        assert(len(units)==1)
        self.unit=next(iter(units))
        self.densities = {int(k):float(list(v.values())[2]) 
                for k,v in data.items()
                if k.isnumeric()}

    def __iter__(self):
        return self.densities.values()

      
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
        self.name = kg_result["name"]
        self.info = kg_result['description']
        self.identifier = kg_result['identifier']
        self.url = "https://search.kg.ebrains.eu/instances/Dataset/{}".format(self.identifier)
        self.modality = kg_result['modality']

        urls = kg_result["files"]
        urls_matching = lambda regex:filter(lambda u:re.match(regex,u),urls)

        # add fingerprint if a url is found
        self._fingerprint_loader = None
        for url in urls_matching(".*_fp[._]"):
            if self._fingerprint_loader is not None:
                logger.warn(f"More than one fingerprint found for {self}")
            self._fingerprint_loader = LazyLoader(url,lambda u:DensityFingerprint(decode_tsv(u)))

        # add any cortical profiles
        self._profile_loaders = {}
        for url in urls_matching(".*_pr[._].*\.tsv"):
            rtype, basename = url.split("/")[-2:]
            if rtype not in basename:
                continue
            if rtype in self._profile_loaders:
                logger.warn(f"More than one profile for '{rtype}' in {self.url}")
            self._profile_loaders[rtype] = LazyLoader(url,lambda u:DensityProfile(decode_tsv(u)))

        # add autoradiograph
        self._autoradiograph_loaders = {}
        img_from_bytes = lambda b:np.array(Image.open(BytesIO(b)))
        for url in urls_matching(".*_ar[._]"):
            rtype, basename = url.split("/")[-2:]
            if rtype not in basename:
                continue
            if rtype in self._autoradiograph_loaders:
                logger.warn(f"More than one autoradiograph for '{rtype}' in {self.url}")
            self._autoradiograph_loaders[rtype] = LazyLoader(url,img_from_bytes)

    @property
    def fingerprint(self):
        if self._fingerprint_loader is None:
            return None
        else:
            return self._fingerprint_loader.data

    @property
    def profiles(self):
        return { rtype:l.data for rtype,l 
            in self._profile_loaders.items()}    
        
    @property
    def autoradiographs(self):
        return { rtype:l.data for rtype,l 
            in self._autoradiograph_loaders.items()}  

    def __str__(self):
        return f"Receptor densites for area{self.regionspec}"

    def table(self):
        """ Outputs a small table of available profiles and autoradiographs. """
        #if len(self.profiles)+len(self.autoradiographs)==0:
                #return style.BOLD+"Receptor density for area {}".format(self.region)+style.END
        return "\n"+"\n".join(
                [style.BOLD+str(self)+style.END] +
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

        from collections import deque

        # plot profiles and fingerprint
        fig = plt.figure(figsize=(8,3))
        plt.subplot(121)
        for _,profile in self.profiles.items():
            plt.plot(list(profile.densities.keys()),np.fromiter(profile.densities.values(),dtype='d'))
        plt.xlabel('Cortical depth (%)')
        plt.ylabel("Receptor density")  
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


class ReceptorQuery(FeatureQuery):

    _FEATURETYPE = ReceptorDistribution

    def __init__(self):
        FeatureQuery.__init__(self)
        kg_query = ebrains.execute_query_by_id('minds', 'core', 'dataset', 'v1.0.0', 'siibra_receptor_densities')
        for kg_result in kg_query['results']:
            region_names = [e['name'] for e in kg_result['region']]
            for region_name in region_names:
                self.register(ReceptorDistribution(region_name,kg_result))

if __name__ == '__main__':
    auth = Authentication.instance()
    auth.set_token('eyJhbGci....')
    extractor = ReceptorQuery()
    print(extractor)
