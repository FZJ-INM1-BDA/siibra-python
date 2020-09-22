from brainscapes import atlases

atlas = atlases.MULTILEVEL_HUMAN_ATLAS
for region in atlas.regiontree.find('hOc1',exact=False):
    print(region)
    atlas.select_region(region)
    atlas.quer
    pool = ReceptorQuery()
    hits = pool.pick_selection(atlas)
    for hit in hits:
        print(hit)

