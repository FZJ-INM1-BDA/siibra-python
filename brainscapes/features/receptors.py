from brainscapes import kg_service
from brainscapes.authentication import Authentication
from brainscapes.features.receptor_data import ReceptorData
from brainscapes.features.feature import RegionalFeature,FeaturePool

__receptor_data_repo = {}


def get_receptor_data_by_region(region_name):
    if len(__receptor_data_repo) == 0:
        _get_receptor_data_for_all_regions()
    return __receptor_data_repo[region_name]


def _get_receptor_data_for_all_regions():
    if len(__receptor_data_repo) == 0:
        kg_result = kg_service.execute_query_by_id('minds', 'core', 'dataset', 'v1.0.0', 'bs_datasets_tmp')
        for region in kg_result['results']:
            region_names = [e['https://schema.hbp.eu/myQuery/name'] for e in region['https://schema.hbp.eu/myQuery/parcellationRegion']]
            for r in region_names:
                __receptor_data_repo[r] = ReceptorData(region)


if __name__ == '__main__':
    auth = Authentication.instance()
    auth.set_token('eyJhbGciOiJSUzI1NiIsImtpZCI6ImJicC1vaWRjIn0.eyJleHAiOjE2MDA3MDEyOTEsInN1YiI6IjMwODExMCIsImF1ZCI6WyJuZXh1cy1rZy1zZWFyY2giXSwiaXNzIjoiaHR0cHM6XC9cL3NlcnZpY2VzLmh1bWFuYnJhaW5wcm9qZWN0LmV1XC9vaWRjXC8iLCJqdGkiOiJhNjUzM2Y4Ni0xMTJjLTRlNTAtYTkxNi01ZTc4MmE2Njg5NzQiLCJpYXQiOjE2MDA2ODY4OTEsImhicF9rZXkiOiJkYTA1ZDc4NGUzMmZjMTM0N2MzNjI5MDIyNDNlYmRjMjdhNDU5MTYyIn0.fIncJMnkMdLXoJGiAYv60fuvmFZ6rrMoE3BfgMsNfY46KeIqoh_t8jTMrO7PYenYZpe75HC3G3QtqzggtFLt8-EbGbnVB4Uo3gSKxf38Dghdov5ZTuHaXCEYFahL3EEJdyYz53Y5WOaCY06EMTxOOjq3RgwUKbzGZ_AA5RiVYJs')
    print(get_receptor_data_by_region('Area 4p (PreCG)'))
    print(get_receptor_data_by_region('Area 4p (PreCG)').profiles)
    print(get_receptor_data_by_region('Area 4p (PreCG)').autoradiographs)
    print(get_receptor_data_by_region('Area 4p (PreCG)').fingerprint)
    # print(__receptor_data_repo)
