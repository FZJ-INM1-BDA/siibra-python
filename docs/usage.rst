==============
Usage examples
==============

To get familiar with ``siibra``, we recommend to checkout the jupyter notebooks in the ``examples/`` subfolder of the source repository. However, here are some code snippets to give you an initial idea.

Retrieving receptor densities for one brain area
------------------------------------------------

.. code:: python

    import siibra as sb
    # NOTE: assumes the client is already authenticated, see above
    atlas = sb.atlases.MULTILEVEL_HUMAN_ATLAS
    atlas.select_region('v1')
    features = atlas.get_features(
        sb.features.modalities.ReceptorDistribution)
    for r in features:
        fig = r.plot(r.region)


Retrieving gene expressions for one brain area
----------------------------------------------

.. code:: python

    import siibra as sb
    from nilearn import plotting
    atlas = sb.atlases.MULTILEVEL_HUMAN_ATLAS
    # request gene expressions from Allen Atlas
    atlas.select_region("v1 left")
    features = atlas.get_features(
        sb.features.modalities.GeneExpression,
        gene=sb.features.gene_names.GABARAPL2 )
    print(features[0])

    # plot
    all_coords = [tuple(g.location) for g in features]
    mask = atlas.get_mask(sb.spaces.MNI_152_ICBM_2009C_NONLINEAR_ASYMMETRIC)
    display = plotting.plot_roi(mask)
    display.add_markers(all_coords,marker_size=5)

