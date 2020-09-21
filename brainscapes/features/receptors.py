from brainscapes import kg_service
from brainscapes.authentication import Authentication
from brainscapes.features.receptor_data import ReceptorData

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
    auth.set_token('eyJhbGciOiJSUzI1NiIsImtpZCI6ImJicC1vaWRjIn0.eyJleHAiOjE2MDA2ODY4NDQsInN1YiI6IjMwODExMCIsImF1ZCI6WyJuZXh1cy1rZy1zZWFyY2giXSwiaXNzIjoiaHR0cHM6XC9cL3NlcnZpY2VzLmh1bWFuYnJhaW5wcm9qZWN0LmV1XC9vaWRjXC8iLCJqdGkiOiI0M2E4YTIzOS0wZGY2LTQ1ZjMtYjdjMi0xODI1M2I4MDQ0NGQiLCJpYXQiOjE2MDA2NzI0NDQsImhicF9rZXkiOiJkYTA1ZDc4NGUzMmZjMTM0N2MzNjI5MDIyNDNlYmRjMjdhNDU5MTYyIn0.jpO_3hDJMGtXcnfwQ-bsL4NXZ22GoaNTE0ozFrlsrHBPdSqm5a_9dfcUtOOUvYd5ixWY8_tNNNoan3ubnTh4hKpVL11TNq-m_mMoeEHq77dDPFzTS5zbAgD2_qdKrK7BG33QgB3nGpj_FTEYc0hk_Qfys0HOQr9npZy79-AmiN0')
    print(get_receptor_data_by_region('Area 4p (PreCG)'))
    print(get_receptor_data_by_region('Area 4p (PreCG)').profiles)
    print(get_receptor_data_by_region('Area 4p (PreCG)').autoradiographs)
    print(get_receptor_data_by_region('Area 4p (PreCG)').fingerprint)
    # print(__receptor_data_repo)
