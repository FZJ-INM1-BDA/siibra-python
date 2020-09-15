#!/usr/bin/env python3

import click
import logging
import brainscapes.atlas as bsa
from brainscapes import preprocessing 
from brainscapes.ontologies import atlases, parcellations, spaces

logging.basicConfig(level=logging.INFO)

def complete_parcellations(ctx, args, incomplete):
    """ auto completion for parcellations """
    return dir(parcellations)

def complete_spaces(ctx, args, incomplete):
    """ auto completion for parcellations """
    return dir(spaces)


@click.group()
@click.option('-p','--parcellation', type=click.STRING, default=None, 
        autocompletion=complete_parcellations,
        help="Specify another than the default parcellation")
@click.option('--cache', default=None, type=click.Path(dir_okay=True),
        help="Local directory for caching downloaded files. If none, a temporary directory will be used.")
@click.pass_context
def brainscapes(ctx,parcellation,cache):
    """ Command line interface to the brainscapes atlas services.
    """
    ctx.obj = bsa.Atlas(cachedir=cache)
    if parcellation is not None:
        if not hasattr(parcellations,parcellation):
            logging.error("No such parcellation available: "+parcellation)
            exit(1)
        ctx.obj.select_parcellation_scheme(getattr(parcellations,parcellation))
    logging.info('Atlas uses parcellation "{}"'.format(ctx.obj.__parcellation__['name']))

@brainscapes.group()
@click.pass_context
def region(ctx):
    """
    Browse the region hierarchy of the selected parcellation.
    """
    pass

@region.command()
@click.argument('searchstring', type=click.STRING)
@click.option('-i','--case-insensitive',is_flag=True,
        help="Ignore case when searching")
@click.pass_context
def search(ctx,searchstring,case_insensitive):
    """
    Search regions from the selected parcellation by substring matching.
    """
    atlas = ctx.obj
    matches = atlas.search_region(searchstring,exact=False)
    for m in matches:
        print(m.name)

@region.command()
@click.pass_context
def list(ctx):
    """
    List all basic regions (leaves of the region hierarchy)
    """
    atlas = ctx.obj
    for region in atlas.regions():
        print(region.name)

@region.command()
@click.pass_context
def hierarchy(ctx):
    """
    Plot the complete region hierarchy of the selected parcellation.
    """
    atlas = ctx.obj
    atlas.regionhierarchy()

@brainscapes.group()
@click.pass_context
def compute(ctx):
    """
    Perform basic computations on brain regions.
    """
    pass

@compute.command()
@click.argument('space', 
        type=click.STRING, 
        autocompletion=complete_spaces)
@click.pass_context
def regionprops(ctx,space):
    """
    Compute basic properties of atlas regions as requested. 
    """

    atlas = ctx.obj
    spaces_obj = getattr(spaces,space)

    # Extract properties of all atlas regions
    lbl_volumes = atlas.get_maps(spaces_obj)
    tpl_volume = atlas.get_template(spaces_obj)
    props = preprocessing.regionprops(lbl_volumes,tpl_volume)

    # Generate commandline report
    for region in atlas.regions():
        is_cortical = region.has_parent('cerebral cortex')
        label = int(region.labelIndex)
        if label not in props.keys():
            print("{n:40.40}  {na[0]:>12.12} {na[0]:>12.12} {na[0]:>12.12}  {na[0]:>10.10}  {na[0]:>10.10}".format(
                n=region.name, na=["N/A"]*5))
            continue
        for prop in props[label]:
            # FIXME this identifies left/right hemisphere labels for
            # Julich-Brain, but is not a general solution
            if prop.labelled_volume_description in region.name:
                print("{n:40.40}  {c[0]:12.1f} {c[1]:12.1f} {c[2]:12.1f}  {v:10.1f}  {s:10.1f}  {i}".format(
                    n=region.name, 
                    c=prop.centroid_mm,
                    v=prop.volume_mm,
                    s=prop.surface_mm,
                    i=is_cortical
                    ))

