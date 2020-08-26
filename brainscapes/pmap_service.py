import gzip

import requests

from brainscapes.roi import Roi
from brainscapes.region import Region

url = 'http://pmap-pmap-service.apps-dev.hbp.eu/multimerge'


def retrieve_probability_map(region, hemisphere, threshold):
    """
    Retrieves a probability map for an area
    :param region: Region object
    :param hemisphere: The Hemisphere. Should be provided through the enum
    :param threshold: Threshold or the probability map
    :return: returns a Roi object containing the nii file if status is OK, otherwise None
    """
    try:
        data = '{ "areas": [{"name": "' + region.name + \
               '","hemisphere": "' + hemisphere + '" }], "threshold": ' + str(threshold) + '}'
        response = requests.post(url, data=data, headers={'Content-Type': 'application/json'})
    except requests.exceptions.RequestException as e:
        response = None
    if response is not None and response.status_code == 200:
        data_nii = gzip.decompress(response.content)
        return Roi(data_nii, region, hemisphere, threshold)
    else:
        print('response', response)
        return None


if __name__ == '__main__':
    roi = retrieve_probability_map(Region('Area-Fp1', 'colin', 'par1'), 'left', 0.2)
    roi.save("zzz")
    roi.save("yyy.nii")
    print(roi)
