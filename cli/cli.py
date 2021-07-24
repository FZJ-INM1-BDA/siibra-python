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
import click
import os


# ---- Autocompletion

def complete_parcellations(ctx, args, incomplete):
    """ auto completion for parcellations """
    return [p for p in dir(siibra.parcellations) 
            if p.startswith(incomplete)]

def complete_spaces(ctx, args, incomplete):
    """ auto completion for spaces """ 
    return [s for s in dir(siibra.spaces) 
            if s.startswith(incomplete)]


# ---- Main command

@click.group()
@click.pass_context
def siibra(ctx):
    """Command line interface to the siibra atlas toolsuite"""
    ctx.obj = {}


# ---- download files

@siibra.group()
@click.option('-o','--outfile',  type=click.STRING, default=None,
        help="Name of output file (suffix will be added if needed)")
@click.pass_context
def get(ctx,outfile):
    """Retrieve different types of atlas data"""
    ctx.obj['outfile'] = outfile
    pass

@get.command()
@click.argument('parcellation',  type=click.STRING, autocompletion=complete_parcellations)
@click.argument('space',  type=click.STRING, autocompletion=complete_spaces)
@click.pass_context
def map(ctx,parcellation,space):
    """Retrieve a parcellation map in the given space"""

    import siibra as sb
    atlas = sb.atlases['human']
    try:
        atlas.select_parcellation(parcellation)
    except IndexError:
        print("Parcellation specification invalid.")
        exit(1)
    parcobj = atlas.selected_parcellation
    try:
        spaceobj = sb.spaces[space]
    except IndexError:
        print("Space specification invalid.")
        exit(1)
    print(f"Loading map of {atlas.selected_parcellation.name} in {spaceobj.name} space.")
    try:
        parcmap = atlas.get_map(spaceobj,sb.MapType.LABELLED)
    except ValueError as e:
        print(str(e)+".")
        exit(1)

    fname = ctx.obj['outfile']
    suffix = ".nii.gz"
    if fname is None:
        fname = click.edit('Output file name', default=f"{parcobj.key}_{spaceobj.key}{suffix}")
    if not fname.endswith(suffix) and not fname.endswith(".nii"):
        fname = f"{os.path.splitext(fname)[0]}{suffix}"

    if len(parcmap)==1:
        # we have a single map
        img = parcmap.fetch()
        print(f"map info: {img.dataobj.dtype} {img.shape}")
        img.to_filename(fname)
        print(f"Output written to {fname}.")
        exit(0)

    # we have multiple maps
    for i,img in enumerate(parcmap.fetchall()):
        fname_ = fname.replace(suffix,f"_{i}{suffix}")
        print(f"map info: {img.dataobj.dtype} {img.shape}")
        img.to_filename(fname_)
        print(f"Output written to {fname_}.")


@get.command()
@click.argument('space', type=click.STRING)
@click.pass_context
def template(ctx,space):
    """Retrieve the template image for a given space"""
    outfile = ctx.obj['outfile']
    import siibra as sb
    spaceobj = sb.spaces[space]
    print(f"Loading template of {spaceobj.name} space.")
    tpl = spaceobj.get_template().fetch()
    suffix = ".nii.gz"
    fname = f"{spaceobj.key}{suffix}" if outfile is None else outfile
    if not fname.endswith(suffix) and not fname.endswith(".nii"):
        fname = f"{os.path.splitext(fname)[0]}{suffix}"
    tpl.to_filename(fname)
    print(f"Output written to {fname}.")


# ---- Searching for things

@siibra.group()
@click.pass_context
def find(ctx):
    """Find atlas concepts by name"""
    pass

@find.command()
@click.argument('region', type=click.STRING)
@click.option('-a','--all', type=click.BOOL, default=False,
        help="Whether to search region in all available parcellations")
@click.option('-p','--parcellation',type=click.STRING, default=None, autocompletion=complete_parcellations)
@click.pass_context
def region(ctx,region,all,parcellation):
    """Find brain regions by name"""
    import siibra as sb
    atlas = sb.atlases['human']
    if parcellation is not None:
        try:
            atlas.select_parcellation(parcellation)
        except IndexError:
            print(f"Cannot select {parcellation} as a parcellation; using default: {atlas.selected_parcellation}")
    print(f"Searching for region '{region}' in {atlas}.")
    matches = atlas.find_regions(region,all_parcellations=all)
    if len(matches)==0:
        print(f"No region found.")
        exit(1)
    for i,m in enumerate(matches):
        print(f"{i:5} - {m}")


@find.command()
@click.argument('region', type=click.STRING)
@click.option('-p','--parcellation',type=click.STRING, default=None, autocompletion=complete_parcellations)
@click.option('-m','--match',type=click.STRING, default=None, help="Filter dataset names by matching them to the given string sequence")
@click.pass_context
def features(ctx,region,parcellation,match):
    """Find data features associated to a brain region"""

    # init siibra
    os.environ["SIIBRA_LOG_LEVEL"]="WARN"
    import siibra as sb
    sb.set_log_level("INFO")
    atlas = sb.atlases['human']
    if parcellation is not None:
        try:
            atlas.select_parcellation(parcellation)
        except IndexError:
            print(f"Cannot select {parcellation} as a parcellation; using default: {atlas.selected_parcellation}")

    # select the requested region
    try:
        matched_region = atlas.select_region(region)
    except ValueError as e:
        print(str(e))
        exit(1)
    try:
        features = atlas.get_features(sb.modalities.EbrainsRegionalDataset)
    except RuntimeError as e:
        print(str(e))
        exit(1)
    if match is None:
        print(f"{len(features)} features found.")
    else:
        N = len(features)
        features = list(filter(lambda f:match in f.name,features))
        print(f"{N} features found for {matched_region.name}, {len(features)} matching the string '{match}'.")
    if len(features)>1:
        for i,m in enumerate(features):
            print(f"{i:5} - {m.name}")
        index = click.prompt('Chooose a feature?', type=click.IntRange(0,len(features)))
    else:
        index = 0
    print(features[index])
    if click.confirm('Open in browser?',default=True):
        click.launch(features[index].url)




# ---- Assign locations

@siibra.group()
@click.pass_context
def assign(ctx):
    """Assign spatial objects to brain regions"""
    pass

@assign.command()
@click.argument('coordinate', type=click.FLOAT, nargs=3)
@click.argument('space',type=click.STRING, autocompletion=complete_spaces)
@click.option('-p','--parcellation',type=click.STRING, default=None, autocompletion=complete_parcellations)
@click.option('--labelled/--probabilistic',is_flag=True, default=True,
    help="Wether to use labelled maps or continuous (e.g. probabilistic) maps to perform the assignment (default:labelled)")
@click.option('-s','--sigma-mm',type=click.FLOAT, default=1.0)
@click.pass_context
def coordinate(ctx,coordinate,space,parcellation,labelled,sigma_mm):
    """Assign a 3D coordinate to brain regions.
    
    Note: To provide negative numbers, add "--" as the first argument after all options, ie. `siibra assign coordinate -- -3.2 4.6 -12.12`
    """
    import siibra as sb
    atlas = sb.atlases['human']
    spaceobj = sb.spaces[space]
    maptype = sb.MapType.LABELLED if labelled else sb.MapType.CONTINUOUS
    print("Using {maptype} type maps.")
    assignments = atlas.assign_coordinates(spaceobj,coordinate,maptype=maptype,sigma_mm=sigma_mm)
    for i,(region,_,scores) in enumerate(assignments[0]):
        if isinstance(scores,dict):
            if i==0:
                headers = "".join(f"{k:>12.12}" for k in scores.keys())
                print(f"{'Scores':40.40} {headers}")
            values = "".join(f"{v:12.2f}" for v in scores.values())
            print(f"{region.name:40.40} {values}")
        else:
            print(region.name)


