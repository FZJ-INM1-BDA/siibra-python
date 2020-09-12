#!/usr/bin/env python3

import click
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
    print("Preprocessing parcellation",parcellation)
    parcellation_obj = getattr(parcellations,parcellation)
    spaces_obj = getattr(spaces,space)
    atlas = bsa.Atlas(cachedir=cache)
    atlas.select_parcellation_scheme(parcellation_obj)
    icbm_map = atlas.get_map(spaces_obj)
    icbm_tpl = atlas.get_template(spaces_obj)
    props = proc.regionprops(icbm_map)
    for region in atlas.regions():
        label = region.labelIndex
        if label not in props.keys():
            print("No properties for label",label)
            continue
        name = region.name
        stored_position = region.position
        computed_position_mm = props[label]['centroid']
        print(name,computed_position_mm)

