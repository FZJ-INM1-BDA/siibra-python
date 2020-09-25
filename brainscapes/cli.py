#!/usr/bin/env python3

import click
import logging
import brainscapes.atlas as bsa
import json
from brainscapes import parcellations, spaces, atlases
from brainscapes.features import modalities
from brainscapes.features.genes import AllenBrainAtlasQuery
from brainscapes.termplot import FontStyles as style

logging.basicConfig(level=logging.INFO)

# ---- Autocompletion functions ----

def complete_parcellations(ctx, args, incomplete):
    """ auto completion for parcellations """
    return [p for p in dir(parcellations) 
            if p.startswith(incomplete)]

def complete_regions(ctx, args, incomplete):
    """ auto completion for parcellations """
    atlas = atlases[0]
    for option in ['-p','--parcellation']:
        if args.count(option):
            pname = args[args.index(option)+1]
            atlas.select_parcellation(pname)
    regions = [c.key 
            for r in atlas.regiontree.iterate()
            for c in r.children ]
    search = [r for r in regions if incomplete.upper() in r]
    return search if len(search)>0 else ""

def complete_spaces(ctx, args, incomplete):
    """ auto completion for parcellations """ 
    return [s for s in dir(spaces) 
            if s.startswith(incomplete)]

def complete_genes(ctx, args, incomplete):
    """ autocompletion for genes """
    if len(incomplete)>0:
        gene_acronyms = AllenBrainAtlasQuery.GENE_NAMES.keys()
        return [a for a in gene_acronyms if a.startswith(incomplete)]
    else:
        return ""

def complete_featuretypes(ctx, args, incomplete):
    """ auto completion for feature types """
    return [m for m in modalities
            if m.startswith(incomplete)]

# ---- Main command ----

@click.group()
@click.option('-p','--parcellation', type=click.STRING, default=None, 
        autocompletion=complete_parcellations,
        help="Specify another than the default parcellation")
@click.option('-s','--space',  type=click.STRING, default=None,
        autocompletion=complete_spaces,
        help="Specify a template space")
@click.pass_context
def brainscapes(ctx,parcellation,space):
    """ Command line interface to the brainscapes atlas services.
    """
    ctx.obj = {}
    ctx.obj['atlas'] = atlas = atlases[0]

    if parcellation is not None:
        try:
            atlas.select_parcellation(parcellation)
        except Exception as e:
            print(str(e))
            logging.error("No such parcellation available: "+parcellation)
            exit(1)
    logging.info('Selected parcellation "{}"'.format(
        ctx.obj['atlas'].selected_parcellation.name))

    if space is None:
        ctx.obj['space'] = atlas.spaces[0]
    else:
        if spaces.obj(space) in atlas.spaces:
            ctx.obj['space'] = spaces.obj(space)
        else:
            logging.error("Space {} is not supported by atlas {}.".format(
                    atlas,space))
            exit(1)
    logging.info('Using template space "{}"'.format(ctx.obj['space']))

# ---- Commands for working with the region hierarchy ----

@brainscapes.group()
@click.pass_context
def hierarchy(ctx):
    """
    Work with the region hierarchy of the selected parcellation.
    """
    pass

@hierarchy.command()
@click.argument('searchstring', type=click.STRING)
@click.option('-i','--case-insensitive',is_flag=True,
        help="Ignore case when searching")
@click.pass_context
def search(ctx,searchstring,case_insensitive):
    """
    Search regions by name.
    """
    atlas = ctx.obj['atlas']
    matches = atlas.regiontree.find(searchstring,exact=False)
    for m in matches:
        print(m.name)

@hierarchy.command()
@click.pass_context
def tree(ctx):
    """
    Print the complete region hierarchy as a tree.
    """
    atlas = ctx.obj['atlas']
    print(atlas.regiontree)

# ---- Commands for retrieving data features ---

@brainscapes.group()
@click.argument('region', type=click.STRING,
        autocompletion=complete_regions )
@click.pass_context
def features(ctx,region):
    """
    Retrieve region specific features.
    """
    atlas = ctx.obj['atlas']
    atlas.select_region(region)

@features.command()
@click.argument('gene', type=click.STRING,
        autocompletion=complete_genes )
@click.pass_context
def gex(ctx,gene):
    """
    Extract gene expressions from the Allen Human Brain Atlas.
    """
    atlas = ctx.obj['atlas']
    hits = atlas.query_data("GeneExpression",gene=gene)
    for hit in hits:
        print(hit)

@features.command()
@click.pass_context
def receptors(ctx):
    """
    Extract receptor distributions from the EBRAINS knowledge graph.
    """
    atlas = ctx.obj['atlas']
    hits = atlas.query_data("ReceptorDistribution")
    for hit in hits:
        print(hit)

@features.command()
@click.pass_context
def connectivity(ctx):
    atlas = ctx.obj['atlas']
    hits = atlas.query_data("ConnectivityProfile")
    for hit in hits:
        print(hit)

@features.command()
@click.pass_context
def props(ctx):
    """
    Return spatial properties of the region
    """
    atlas,space = [ctx.obj[t] for t in ['atlas','space']]
    props = atlas.regionprops(space)
    print(style.BOLD)
    print("Region properties of {}".format(atlas.selected_region))
    print(style.END)
    print(props)

