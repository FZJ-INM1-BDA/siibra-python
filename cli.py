#!/usr/bin/env python3

import click
import logging
import brainscapes.atlas as bsa
import brainscapes.preprocessing as proc
from brainscapes.ontologies import atlases, parcellations, spaces

@click.group()
def brainscapes():
    """ Command line interface to the brainscapes atlas services.
    """
    pass    

def complete_parcellations(ctx, args, incomplete):
    """ auto completion for parcellations """
    return dir(parcellations)

def complete_spaces(ctx, args, incomplete):
    """ auto completion for parcellations """
    return dir(spaces)

@brainscapes.command()
@click.argument('parcellation', 
        type=click.STRING, 
        autocompletion=complete_parcellations)
@click.argument('space', 
        type=click.STRING, 
        autocompletion=complete_spaces)
@click.option('--cache', default=None, type=click.Path(dir_okay=True),
        help="Local directory for caching downloaded files. If none, a temporary directory will be used.")
@click.pass_context
def regionprops(ctx,parcellation,space,cache):

    parcellation_obj = getattr(parcellations,parcellation)
    spaces_obj = getattr(spaces,space)

    atlas = bsa.Atlas(cachedir=cache)
    atlas.select_parcellation_scheme(parcellation_obj)
    lbl_volume = atlas.get_map(spaces_obj)
    tpl_volume = atlas.get_template(spaces_obj)
    props = proc.regionprops(lbl_volume,tpl_volume)
    for region in atlas.regions():
        print(region.name)
        label = int(region.labelIndex)
        if label not in props.keys():
            logging.warn("No properties for label {}".format(label))
            continue
        stored_position = region.position
        for prop in props[label]:
            print("{d[0]:12.1f} {d[1]:12.1f} {d[2]:12.1f}   {c[0]:12.1f} {c[1]:12.1f} {c[2]:12.1f}".format(
                d=stored_position,
                c=prop.centroid_mm))

