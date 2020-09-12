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

@brainscapes.command()
@click.argument('parcellation', type=click.STRING, autocompletion=complete_parcellations)
@click.pass_context
def preprocess(ctx,parcellation):
    print("Preprocessing parcellation",parcellation)

