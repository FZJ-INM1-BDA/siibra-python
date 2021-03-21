#!/usr/bin/env python3

# Copyright 2018-2020 Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# disable logging during setup
from brainscapes import logger
logger.setLevel('DEBUG')

import click
import brainscapes as bs
from brainscapes.termplot import FontStyles as style

# re-enable logging for execution of cli commands
logger.setLevel('DEBUG')

# ---- Autocompletion functions ----

def complete_parcellations(ctx, args, incomplete):
    """ auto completion for parcellations """
    return [p for p in dir(bs.parcellations) 
            if p.startswith(incomplete)]

def complete_regions(ctx, args, incomplete):
    """ auto completion for regions """
    atlas = bs.atlases[0]
    for option in ['-p','--parcellation']:
        if args.count(option):
            pname = args[args.index(option)+1]
            atlas.select_parcellation(pname)
    regions = [c.key 
            for r in atlas.selected_parcellation.regions.iterate()
            for c in r.children ]
    print(regions)
    search = [r for r in regions if incomplete.upper() in r]
    return search if len(search)>0 else ""

def complete_spaces(ctx, args, incomplete):
    """ auto completion for spaces """ 
    return [s for s in dir(bs.spaces) 
            if s.startswith(incomplete)]

def complete_genes(ctx, args, incomplete):
    """ autocompletion for genes """
    if len(incomplete)>0:
        return [a for a in dir(bs.features.gene_names)
                if a.startswith(incomplete)]
    else:
        return ""

def complete_regional_modalities(ctx, args, incomplete):
    """ auto completion for regional feature types """
    return [m for m in bs.features.modalities
            if m.startswith(incomplete) 
            and bs.features.feature.GlobalFeature 
            not in bs.features.classes[m].__bases__ ] + ['RegionProps']

def complete_global_modalities(ctx, args, incomplete):
    """ auto completion for global feature types """
    return [m for m in bs.features.modalities
            if m.startswith(incomplete) 
            and bs.features.feature.GlobalFeature 
            in bs.features.classes[m].__bases__ ]

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
    ctx.obj['atlas'] = atlas = bs.atlases[0]

    if parcellation is not None:
        try:
            atlas.select_parcellation(parcellation)
        except Exception as e:
            print(str(e))
            bs.logger.error("No such parcellation available: "+parcellation)
            exit(1)

    if space is None:
        ctx.obj['space'] = atlas.spaces[0]
    else:
        if bs.spaces.obj(space) in atlas.spaces:
            ctx.obj['space'] = bs.spaces.obj(space)
        else:
            bs.logger.error("Space {} is not supported by atlas {}.".format(
                    atlas,space))
            exit(1)
    bs.logger.info('Using template space "{}"'.format(ctx.obj['space']))

# ---- Commands for working with the region hierarchy ----

@brainscapes.group()
@click.pass_context
def regions(ctx):
    """
    Work with the region hierarchy of the selected parcellation.
    """
    pass

@regions.command()
@click.argument('searchstring', type=click.STRING)
@click.option('-i','--case-insensitive',is_flag=True,
        help="Ignore case when searching")
@click.pass_context
def search(ctx,searchstring,case_insensitive):
    """
    Search regions by name.
    """
    atlas = ctx.obj['atlas']
    matches = atlas.selected_parcellation.regions.find(searchstring)
    for m in matches:
        print(m.name)

@regions.command()
@click.pass_context
def tree(ctx):
    """
    Print the complete region hierarchy as a tree.
    """
    atlas = ctx.obj['atlas']
    print(repr(atlas.selected_parcellation.regions))

# ---- Commands for retrieving data features ---

@brainscapes.command()
@click.argument('modality', type=click.STRING, 
        autocompletion=complete_regional_modalities)
@click.argument('region', type=click.STRING,
        autocompletion=complete_regions )
@click.option('-g','--gene', type=click.STRING, default=None,
        autocompletion=complete_genes )
@click.pass_context
def features(ctx,modality,region,gene):
    """
    Retrieve region specific data features.
    """
    atlas = ctx.obj['atlas']
    atlas.select_region(region)

    header = ""
    if modality=='RegionProps':
        space = ctx.obj['space']
        results = [atlas.regionprops(space)]
        header = style.BOLD+"Region properties"+style.END
    elif modality=='GeneExpression':
        if gene is None:
            print("You need to specify a gene with the -g option when looking up gene expressions.")
            return 1
        results = atlas.query_data(modality,gene=gene)
        header = style.BOLD+"Gene Expressions"+style.END
    else:
        results = atlas.query_data(modality)

    if len(results)>0:
        print(header)
    else:
        print('No "{}" features found for "{}".'.format(
            modality,region))
    for result in results:
        print(result)

@brainscapes.command()
@click.argument('modality', type=click.STRING, 
        autocompletion=complete_global_modalities)
@click.pass_context
def globals(ctx,modality):
    """
    Retrieve global data features.
    """
    atlas = ctx.obj['atlas']
    features = atlas.query_data(modality)
    for feature in features:
        print(feature)

