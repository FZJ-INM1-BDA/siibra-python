from brainscapes import atlases
from brainscapes.features import pools

atlas = atlases.MULTILEVEL_HUMAN_ATLAS
for region in atlas.regiontree.find('hOc1',exact=False):
    atlas.select_region(region)
    hits = atlas.query_data("ReceptorDistribution")
    for hit in hits:
        print(hit)

