#!/usr/bin/env python3

# Copyright 2018-2024
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

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
from pkg_resources import iter_entry_points
from click_plugins import with_plugins
import os

from siibra.retrieval.requests import SiibraHttpRequestError
from siibra.core.concept import get_registry
# ---- Autocompletion


class SpaceName(click.ParamType):
    name = "space"

    def shell_complete(self, ctx, param, incomplete):
        print(incomplete)
        return [
            click.CompletionItem(s.key)
            for s in get_registry('space')
            if s.key.startswith(incomplete)
        ]


class ParcellationName(click.ParamType):
    name = "parcellation"

    def shell_complete(self, ctx, param, incomplete):
        return [
            click.CompletionItem(s.key)
            for s in get_registry('parcellation')
            if s.key.startswith(incomplete)
        ]


# ---- Main command


@with_plugins(iter_entry_points('siibra_cli.plugins'))
@click.group()
@click.pass_context
@click.option(
    "-s",
    "--species",
    type=click.STRING,
    default="human",
    help="Species (human, rat, mouse)",
)
def cli(ctx, species):
    """Command line interface to the siibra atlas toolsuite"""
    ctx.obj = {"species": species}


# ---- download files


@cli.group()
@click.option(
    "-o",
    "--outfile",
    type=click.STRING,
    default=None,
    help="Name of output file (suffix will be added if needed)",
)
@click.pass_context
def get(ctx, outfile):
    """Retrieve different types of atlas data"""
    ctx.obj["outfile"] = outfile
    pass


@get.command()
@click.argument("parcellation", type=ParcellationName())
@click.argument("space", type=SpaceName())
@click.pass_context
def map(ctx, parcellation, space):
    """Retrieve a parcellation map in the given space"""
    import siibra as siibra

    atlas = siibra.atlases[ctx.obj["species"]]
    siibra.logger.info(f"Using atlas '{atlas.name}'.")
    try:
        parcobj = atlas.get_parcellation(parcellation)
    except IndexError:
        click.echo("Parcellation specification invalid.")
        exit(1)

    try:
        spaceobj = atlas.get_space(space)
    except IndexError:
        click.echo("Space specification invalid.")
        exit(1)
    click.echo(f"Loading map of {parcobj.name} in {spaceobj.name} space.")
    try:
        parcmap = atlas.get_map(
            space=spaceobj, parcellation=parcobj, maptype=siibra.MapType.LABELLED
        )
    except ValueError as e:
        click.echo(str(e) + ".")
        exit(1)

    fname = ctx.obj["outfile"]
    suffix = ".nii.gz"
    if fname is None:
        fname = click.prompt(
            "Output file name", f"{parcobj.key}_{spaceobj.key}{suffix}"
        )
    if not fname.endswith(suffix) and not fname.endswith(".nii"):
        fname = f"{os.path.splitext(fname)[0]}{suffix}"

    if len(parcmap) == 1:
        # we have a single map
        img = parcmap.fetch()
        img.to_filename(fname)
        exit(0)
    else:
        for i, img in enumerate(parcmap.fetch_iter()):
            fname_ = fname.replace(suffix, f"_{i}{suffix}")
            img.to_filename(fname_)
            click.echo(f"File {i+1} of {len(parcmap)} written to '{fname_}'.")


@get.command()
@click.argument("space", type=click.STRING)
@click.pass_context
def template(ctx, space):
    """Retrieve the template image for a given space"""
    outfile = ctx.obj["outfile"]
    import siibra

    atlas = siibra.atlases[ctx.obj["species"]]
    siibra.logger.info(f"Using atlas '{atlas.name}'.")
    spaceobj = atlas.get_space(space)
    tpl = atlas.get_template(spaceobj)
    click.echo(f"Loading template of {spaceobj.name} space.")
    img = tpl.fetch()
    suffix = ".nii.gz"
    fname = f"{spaceobj.key}{suffix}" if outfile is None else outfile
    if not fname.endswith(suffix) and not fname.endswith(".nii"):
        fname = f"{os.path.splitext(fname)[0]}{suffix}"
    img.to_filename(fname)
    click.echo(f"Output written to {fname}.")


@get.command()
def ebrainstoken():
    """Retrieve a parcellation map in the given space"""
    import siibra

    try:
        siibra.fetch_ebrains_token()
        print(siibra.EbrainsRequest._KG_API_TOKEN)
    except SiibraHttpRequestError:
        exit(1)


# ---- Searching for things


@cli.group()
@click.pass_context
def find(ctx):
    """Find atlas concepts by name"""
    pass


@find.command()
@click.argument("region", type=click.STRING, nargs=-1)
@click.option(
    "-p",
    "--parcellation",
    type=ParcellationName(),
    default=None,
)
@click.option(
    "--showtree/--no-tree",
    default=False,
)
@click.pass_context
def region(ctx, region: str, parcellation: str = None, showtree: bool = False):
    """Find brain regions by name"""
    import siibra

    atlas = siibra.atlases[ctx.obj["species"]]
    siibra.logger.info(f"Using atlas '{atlas.name}'.")
    regionspec = " ".join(region)
    if parcellation is None:
        click.echo(f"Searching for region '{regionspec}' in all parcellations.")
        matches = atlas.find_regions(regionspec)
    else:
        parcobj = atlas.get_parcellation(parcellation)
        click.echo(f"Searching for region '{regionspec}' in {parcobj.name}.")
        matches = parcobj.find(regionspec)

    if len(matches) == 0:
        click.echo(f"No region found using the specification {regionspec}.")
        exit(1)
    for i, m in enumerate(matches):
        txt = m.tree2str() if showtree else m.name
        if parcellation is None:
            click.echo(f"{i:5} | {m.parcellation.name:30.30} | {txt}")
        else:
            click.echo(f"{i:5}  {txt}")


@find.command()
@click.argument("region", type=click.STRING, nargs=-1)
@click.option(
    "-p",
    "--parcellation",
    type=ParcellationName(),
    default=None,
)
@click.option(
    "-m",
    "--match",
    type=click.STRING,
    default=None,
    help="Filter dataset names by matching them to the given string sequence",
)
@click.pass_context
def features(ctx, region, parcellation, match):
    """Find data features associated to a brain region"""

    # init siibra
    os.environ["SIIBRA_LOG_LEVEL"] = "WARN"
    import siibra

    siibra.commons.set_log_level("INFO")
    atlas = siibra.atlases[ctx.obj["species"]]
    siibra.logger.info(f"Using atlas '{atlas.name}'.")
    parcobj = atlas.get_parcellation(parcellation)

    regionspec = " ".join(region)
    try:
        regionobj = atlas.get_region(regionspec, parcellation=parcobj)
    except ValueError:
        click.echo(
            f'Cannot decode region specification "{regionspec}" for {parcobj.name}.'
        )
        exit(1)

    features = siibra.get_features(regionobj, "ebrains")
    if match is not None:
        N = len(features)
        features = list(filter(lambda f: match in f.name, features))
        click.echo(
            f"{N} features found for {regionobj.name}, {len(features)} matching the string '{match}'."
        )

    if len(features) == 0:
        click.echo(f"No features found for {regionobj.name} in {parcobj.name}")
        exit(1)

    if len(features) > 1:
        from simple_term_menu import TerminalMenu

        menu = TerminalMenu(f.name for f in features)
        index = menu.show()
    else:
        index = 0

    click.echo(features[index])
    if click.confirm("Open in browser?", default=True):
        click.launch(features[index].url)


# ---- Assign locations


@with_plugins(iter_entry_points('siibra_cli.assignment_plugins'))
@cli.group()
@click.pass_context
def assign(ctx):
    """Assign spatial objects to brain regions"""
    pass


@assign.command()
@click.argument("coordinate", type=click.FLOAT, nargs=3)
@click.argument("space", type=SpaceName())
@click.option(
    "-p",
    "--parcellation",
    type=ParcellationName(),
    default=None,
)
@click.option(
    "--labelled/--probabilistic",
    is_flag=True,
    default=True,
    help="Wether to use labelled maps or continuous (e.g. probabilistic) maps to perform the assignment (default:labelled)",
)
@click.option("-s", "--sigma-mm", type=click.FLOAT, default=1.0)
@click.pass_context
def coordinate(ctx, coordinate, space, parcellation, labelled, sigma_mm):
    """Assign a 3D coordinate to brain regions.

    Note: To provide negative numbers, add "--" as the first argument after all options, ie. `siibra assign coordinate -- -3.2 4.6 -12.12`
    """
    import siibra

    atlas = siibra.atlases[ctx.obj["species"]]
    siibra.logger.info(f"Using atlas '{atlas.name}'.")
    parcobj = atlas.get_parcellation(parcellation)
    maptype = siibra.MapType.LABELLED if labelled else siibra.MapType.STATISTICAL
    requested_space = atlas.get_space(space)
    location = siibra.Point(coordinate, space=requested_space)

    assignments = []
    for spaceobj in [requested_space] + list(atlas.spaces - requested_space):
        try:
            parcmap = atlas.get_map(
                parcellation=parcobj, space=spaceobj, maptype=maptype
            )
            new = parcmap.assign_coordinates(location, sigma_mm=sigma_mm)
            assignments.extend(new[0])
        except (RuntimeError, ValueError):
            continue

        if len(assignments) > 0 and spaceobj == requested_space:
            break

    if len(assignments) == 0:
        click.echo(f"No assignment could be made to {coordinate}.")
        exit(1)

    for i, (region, _, scores) in enumerate(assignments):
        if isinstance(scores, dict):
            if i == 0:
                headers = "".join(f"{k:>12.12}" for k in scores.keys())
                click.echo(f"{'Scores':40.40} {headers}")
            values = "".join(f"{v:12.2f}" for v in scores.values())
            click.echo(f"{region.name:40.40} {values}")
        else:
            click.echo(region.name)
