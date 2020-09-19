import requests
from sys import argv
import json
import numpy as np
import io
import PIL.Image as Image
import pandas as pd
from os import path
import logging

access_token='eyJhbGciOiJSUzI1NiIsImtpZCI6ImJicC1vaWRjIn0.eyJleHAiOjE2MDAzNjczOTAsInN1YiI6IjI1NTIzMCIsImF1ZCI6WyIzMjMxNDU3My1hMjQ1LTRiNWEtYjM3MS0yZjE1YWNjNzkxYmEiXSwiaXNzIjoiaHR0cHM6XC9cL3NlcnZpY2VzLmh1bWFuYnJhaW5wcm9qZWN0LmV1XC9vaWRjXC8iLCJqdGkiOiI2YTBmNjU3NS05MWI3LTQ5NmUtODFlZi0zNGM3NWY1NmU0NjciLCJpYXQiOjE2MDAzNTI5OTAsImhicF9rZXkiOiJlZDQ3ZWE3NjgxYTVjMTdjYzIyZjM1MmYxODVlZWZjYWYwOWI0MDUyIn0.ql_u3OXNuIT0nNGtmZimXjXs0TQDHIcbJs0i5Hbp3MVgKDwTlzBLFGm5OcMpk8fJz98UBOnoL-NQChxThDVoIKnD5OPnHXyQoE-WFRRci_FJmNWGsByndIr6Uc48qzip14Fu61DaqwkX52h1wh-DS-hMrlHY58hy1L2Sx-Uaa_I'
namespace='Minds'
cls_version='core/dataset/v1.0.0'

class ReceptorData:

    # data rows
    profiles = {}
    # images
    autoradiographs = {}
    # math symbols for receptors


    def __init__(self,kg_response):

        self.regions = [e['https://schema.hbp.eu/myQuery/name'] 
            for e in kg_response['https://schema.hbp.eu/myQuery/parcellationRegion']]

        for fname in kg_response['https://schema.hbp.eu/myQuery/v1.0.0']:

            if 'receptors.tsv' in fname:
                bytestream = io.BytesIO(requests.get(fname).content)
                self.__symbols = pd.read_csv(bytestream,sep='\t')
                self.receptor_label = {r._1:r._2 
                        for r in self.__symbols.itertuples()}

            # Receive cortical profiles, if any
            if '_pr_' in fname:
                suffix = path.splitext(fname)[-1] 
                if suffix == '.tsv':
                    receptor_type,basename = fname.split("/")[-2:]
                    if receptor_type in basename:
                        print("Cortical profile of {}: {}".format(
                            receptor_type,fname))
                        bytestream = io.BytesIO(requests.get(fname).content)
                        self.profiles[receptor_type] = pd.read_csv(bytestream,sep='\t')
                else:
                    logging.debug('Expected .tsv for profile, got {}: {}'.format(suffix,fname))

            if '_ar_' in fname:
                receptor_type,basename = fname.split("/")[-2:]
                if receptor_type in basename:
                    print("Autoradiograph of {}: {}".format(
                            receptor_type,fname))
                    bytestream = requests.get(fname).content
                    self.autoradiographs[receptor_type] = Image.open(io.BytesIO(bytestream))

            if '_fp_' in fname:
                print("Fingerprint: {}".format(fname))
                bytestream = io.BytesIO(requests.get(fname).content)
                self.fingerprint = pd.read_csv(bytestream,sep='\t')


if __name__ == "__main__":

    query_name = None if len(argv)<2 else argv[1]
    if query_name is None:
        print("USAGE:",argv[0],"query_name [query_spec.json]")
        exit(1)
    query_spec = None if len(argv)<3 else argv[2]

    if query_spec is not None:
        
        # If an argument is given, assume it is a query defintion in json
        # format. Build the query, and store it in the KG.

        url = "https://kg.humanbrainproject.eu/query/{}/{}/{}".format(
                namespace.lower(), 
                cls_version,
                query_name)
        print(url)
        r = requests.put( url, 
                data = open(query_spec,'r'),
                headers={
                    'Content-Type':'application/json',
                    'Authorization': 'Bearer {}'.format(access_token)})
        if r.ok:
            print('Successfully stored the query at %s ' % url)
        else:
            print(r)
            print('Problem with "put" protocol on url: %s ' % url )

    else: 

        # If no spec is given, run the query. 

        url = "https://kg.humanbrainproject.eu/query/{}/{}/{}/instances".format(
                namespace.lower(),cls_version,query_name)
        r = requests.get(
                url,
                headers={
                    'Content-Type':'application/json',
                    'Authorization': 'Bearer {}'.format(access_token)})
        if r.ok:
            print('Successfully issued the query at %s ' % url)
            results = json.loads(r.content)
            for r in results['results']:
                for k,v in r.items():
                    print(k,v)
                try:
                    receptors = ReceptorData(r)
                    for name,dataframe in receptors.profiles.items():
                        print(name,receptors.receptor_label[name])
                    continue
                except Exception as e:
                    logging.info('Could not generate receptor data from the response.')
                    print(str(e))
                    continue
        else:
            print('Problem with "get" protocol on url: %s ' % url )

