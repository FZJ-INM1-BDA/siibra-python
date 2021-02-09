import os
import brainscapes as bs

from test.get_token import get_token

bs.logger.setLevel("DEBUG")

token = get_token()
print(token)
os.environ['HBP_AUTH_TOKEN'] = token["access_token"]

# find available cortical receptor profiles
receptor_extractor = bs.features.extractor_types.ReceptorDistribution[0]()
profiles = {}
symbols = {}
for feature in receptor_extractor.features:
    symbols = {**symbols, **feature.symbols} # merge symbol definitions of receptors
    print(feature.fingerprint)
    break
    for rtype,profile in feature.profiles.items():
        assert(rtype in symbols)
        profiles[feature.region,rtype] = profile
regions = {region for region,rtype in profiles.keys()}
rtypes = {rtype for region,rtype in profiles.keys()}

