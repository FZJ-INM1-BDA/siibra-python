#!/usr/bin/env python3

import click
import logging
import brainscapes.atlas as bsa
import json
from brainscapes import parcellations, spaces, atlases
from brainscapes.features import modalities
from brainscapes.features.genes import AllenBrainAtlasQuery

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

def complete_featuretypes(ctx, args, incomplete):
    """ auto completion for feature types """
    return [m for m in modalities
            if m.startswith(incomplete)]

# ---- Main command ----

@click.group()
@click.option('-p','--parcellation', type=click.STRING, default=None, 
        autocompletion=complete_parcellations,
        help="Specify another than the default parcellation")
@click.pass_context
def brainscapes(ctx,parcellation):
    """ Command line interface to the brainscapes atlas services.
    """
    ctx.obj = atlases[0]
    if parcellation is not None:
        try:
            ctx.obj.select_parcellation(parcellation)
        except Exception as e:
            print(str(e))
            logging.error("No such parcellation available: "+parcellation)
            exit(1)
    logging.info('Atlas uses parcellation "{}"'.format(
        ctx.obj.selected_parcellation.name))

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
    atlas = ctx.obj
    matches = atlas.regiontree.find(searchstring,exact=False)
    for m in matches:
        print(m.name)

@hierarchy.command()
@click.pass_context
def show(ctx):
    """
    Print the complete region hierarchy.
    """
    atlas = ctx.obj
    print(atlas.regiontree)

@brainscapes.group()
@click.argument('region', type=click.STRING,
        autocompletion=complete_regions )
@click.pass_context
def features(ctx,region):
    """
    Browse the region hierarchy of the selected parcellation.
    """
    atlas = ctx.obj
    atlas.select_region(region)

@features.command()
@click.argument('gene', type=click.STRING,
        autocompletion=complete_genes )
@click.pass_context
def gex(ctx,gene):
    """
    Extract gene expressions from the Allen Human Brain Atlas.
    """
    atlas = ctx.obj
    hits = atlas.query_data("GeneExpression",gene=gene)
    for hit in hits:
        print(hit)

@features.command()
@click.pass_context
def receptors(ctx):
    """
    Extract receptor distributions from the EBRAINS knowledge graph.
    """
    atlas = ctx.obj
    hits = atlas.query_data("ReceptorDistribution")
    for hit in hits:
        print(hit)

@features.command()
@click.pass_context
def connectivity(ctx):
    atlas = ctx.obj
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

    atlas = ctx.obj
    region = atlas.selected_region
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

