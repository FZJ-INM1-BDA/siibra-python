import requests
from sys import argv
import json
import numpy as np
import io
import PIL.Image as Image
import pandas as pd
from os import path
import logging

access_token='eyJhbGciOiJSUzI1NiIsImtpZCI6ImJicC1vaWRjIn0.eyJleHAiOjE2MDAzNDA4ODcsInN1YiI6IjI1NTIzMCIsImF1ZCI6WyIzMjMxNDU3My1hMjQ1LTRiNWEtYjM3MS0yZjE1YWNjNzkxYmEiXSwiaXNzIjoiaHR0cHM6XC9cL3NlcnZpY2VzLmh1bWFuYnJhaW5wcm9qZWN0LmV1XC9vaWRjXC8iLCJqdGkiOiI1NzFjNGY4Yi1hMzdmLTQ3NzktODkxMS00NGE1Y2JkZTQ3NzMiLCJpYXQiOjE2MDAzMjY0ODcsImhicF9rZXkiOiJlZDQ3ZWE3NjgxYTVjMTdjYzIyZjM1MmYxODVlZWZjYWYwOWI0MDUyIn0.h8GMpdlsvjPe11FwdaDbBAgdzNwib-Z7yT2HfE4_befS4yydRzs2OOzxxi6g1B1yBXwDEqZQDpD-993VufKFMPGqvHt7MSk_777K5bgZrPBocrO8091WzMtwfIgaYrdZP-9U3Jc4810Cr6wn6epcbedAyY4H8f3_Lbk5jqDAmrk'


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

    #
    # try uploading a query
    #
    namespace='Minds'
    cls_version='core/dataset/v1.0.0'
    name='bs_datasets'
    upload = False

    if len(argv)>1:
        
        # If an argument is given, assume it is a query defintion in json
        # format. Build the query, and store it in the KG.

        spec=argv[1]

        url = "https://kg.humanbrainproject.eu/query/{}/{}/{}".format(
                namespace.lower(), 
                cls_version,
                name)
        print(url)
        r = requests.put(
                url,
                data = open(spec,'r'),
                headers={
                    'Content-Type':'application/json',
                    'Authorization': 'Bearer {}'.format(access_token)})

        if r.ok:
            print('Successfully stored the query at %s ' % url)
        else:
            print(r)
            print('Problem with "put" protocol on url: %s ' % url )


    else: 

        # If no argument is given, run the query stored in the KG..

        url = "https://kg.humanbrainproject.eu/query/{}/{}/{}/instances".format(
                namespace.lower(),cls_version,name)
        r = requests.get(
                url,
                headers={
                    'Content-Type':'application/json',
                    'Authorization': 'Bearer {}'.format(access_token)})
        if r.ok:
            print('Successfully issued the query at %s ' % url)
            results = json.loads(r.content)
            for r in results['results']:
                receptors = ReceptorData(r)
                for name,dataframe in receptors.profiles.items():
                    print(name,receptors.receptor_label[name])
                #print("Profiles found:",recobj.profiles.keys())
                #print("Autoradiographs found:",recobj.autoradiographs.keys())
        else:
            print('Problem with "get" protocol on url: %s ' % url )

