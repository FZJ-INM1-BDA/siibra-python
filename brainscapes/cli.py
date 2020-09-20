#!/usr/bin/env python3

import click
import logging
import brainscapes.atlas as bsa
import json
from brainscapes import preprocessing, NAME2IDENTIFIER
from brainscapes.features.genes import AllenBrainAtlasQuery
from brainscapes.ontologies import atlases, parcellations, spaces

logging.basicConfig(level=logging.INFO)

def complete_parcellations(ctx, args, incomplete):
    """ auto completion for parcellations """
    return [p for p in dir(parcellations) 
            if p.startswith(incomplete)]

def complete_regions(ctx, args, incomplete):
    """ auto completion for parcellations """
    atlas = bsa.Atlas()
    parcellation = atlas.__parcellation__
    for option in ['-p','--parcellation']:
        if args.count(option):
            pname = args[args.index(option)+1]
            parcellation = parcellations[pname]
            atlas.select_parcellation(parcellation)
    regions = [NAME2IDENTIFIER(r.name) for r in atlas.regiontree.iterate()]
    search = [r for r in regions if incomplete.upper() in r]
    return search if len(search)>0 else ""

def complete_spaces(ctx, args, incomplete):
    """ auto completion for parcellations """ 
    return [s for s in dir(spaces) 
            if s.startswith(incomplete)]

def complete_genes(ctx, args, incomplete):
    """ autocompletion for genes """
    gene_acronyms = AllenBrainAtlasQuery.GENE_NAMES.keys()
    return [a for a in gene_acronyms if a.startswith(incomplete)]

@click.group()
@click.option('-p','--parcellation', type=click.STRING, default=None, 
        autocompletion=complete_parcellations,
        help="Specify another than the default parcellation")
@click.pass_context
def brainscapes(ctx,parcellation):
    """ Command line interface to the brainscapes atlas services.
    """
    ctx.obj = {'atlas' : bsa.Atlas()}
    if parcellation is not None:
        try:
            ctx.obj['atlas'].select_parcellation(parcellations[parcellation])
        except Exception as e:
            print(str(e))
            logging.error("No such parcellation available: "+parcellation)
            exit(1)
    logging.info('Atlas uses parcellation "{}"'.format(
        ctx.obj['atlas'].__parcellation__['name']))

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
def show(ctx):
    """
    Print the complete region hierarchy.
    """
    atlas = ctx.obj['atlas']
    atlas.regiontree.print_hierarchy()

@brainscapes.group()
@click.argument('region', type=click.STRING,
        autocompletion=complete_regions )
@click.argument('space', type=click.STRING,
        autocompletion=complete_spaces )
@click.pass_context
def features(ctx,region,space):
    """
    Browse the region hierarchy of the selected parcellation.
    """
    atlas = ctx.obj['atlas']
    atlas.select_region(region)
    ctx.obj['space'] = spaces[space]

@features.command()
@click.argument('gene', type=click.STRING,
        autocompletion=complete_genes )
@click.pass_context
def gex(ctx,gene):
    """
    Extract gene expressions from the Allen Human Brain Atlas.
    """
    atlas = ctx.obj['atlas']
    space = ctx.obj['space']
    featurepool = AllenBrainAtlasQuery(gene)
    mask = atlas.get_mask(space)
    features = featurepool.inside_mask(space,mask)

@features.command()
@click.pass_context
def connectivity(ctx):
    atlas = ctx.obj['atlas']
    sources = atlas.connectivity_sources()
    print("Available sources:",sources)
    print(atlas.connectivity_matrix(sources[0]))

@features.command()
@click.argument('space', 
        type=click.STRING, 
        autocompletion=complete_spaces)
@click.pass_context
def props(ctx,space):
    """
    Compute basic properties of atlas regions as requested. 
    """

    atlas = ctx.obj['atlas']
    region = atlas.selection
    spaces_obj = spaces[space]

    # Extract properties of all atlas regions
    lbl_volumes = atlas.get_maps(spaces_obj)
    tpl_volume = atlas.get_template(spaces_obj)
    props = preprocessing.regionprops(lbl_volumes,tpl_volume)

    # Generate commandline report
    is_cortical = region.has_parent('cerebral cortex')
    label = int(region.labelIndex)
    if label not in props.keys():
        print("{n:40.40}  {na[0]:>12.12} {na[0]:>12.12} {na[0]:>12.12}  {na[0]:>10.10}  {na[0]:>10.10}".format(
            n=region.name, na=["N/A"]*5))
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

