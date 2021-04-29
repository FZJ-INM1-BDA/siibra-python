==============
Usage examples
==============

To get familiar with ``siibra``, we recommend to checkout the jupyter notebooks in the ``examples/`` subfolder of the source repository. However, here are some code snippets to give you an initial idea.

Retrieving receptor densities for one brain area
------------------------------------------------

.. code:: python

    import siibra
    # NOTE: assumes the client is already authenticated, see above
    atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
    atlas.select_region('v1')
    features = atlas.get_features(
        siibra.features.modalities.ReceptorDistribution)
    for r in features:
        fig = r.plot(r.region)


Retrieving gene expressions for one brain area
----------------------------------------------

.. code:: python

    import siibra
    from nilearn import plotting
    atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
    # request gene expressions from Allen Atlas
    atlas.select_region("v1 left")
    features = atlas.get_features(
        siibra.features.modalities.GeneExpression,
        gene=siibra.features.gene_names.GABARAPL2 )
    print(features[0])

    # plot
    all_coords = [tuple(g.location) for g in features]
    mask = atlas.build_mask(siibra.spaces.MNI152_2009C_NONL_ASYM)
    display = plotting.plot_roi(mask)
    display.add_markers(all_coords,marker_size=5)

