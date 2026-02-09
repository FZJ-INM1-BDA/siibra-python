# Copyright 2018-2021
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

import math
import re
from urllib.parse import urlparse
from typing import Callable, Union
from zipfile import ZipFile
import json

from .. import anchor as _anchor
from . import cortical_profile, tabular

from ...commons import PolyLine, logger, create_key
from ...core import region as _region
from ...retrieval import requests
from ...retrieval.repositories import RepositoryConnector, ZipfileConnector
from ...retrieval.cache import CACHE

from skimage.draw import polygon
from skimage.transform import resize
from io import BytesIO
from textwrap import wrap
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class DFWithMeta(pd.DataFrame):
    _metadata = ['meta']
    @property
    def _constructor(self):
        return DFWithMeta


class MarmosetCalbindinDensityProfile(
    tabular.Tabular,
    configuration_folder="features/tabular/corticalprofiles/marmosetcelldensity",
    category='cellular'
):

    DESCRIPTION = (
        "Marmoset Calbindin Cell Density Profile"
    )

    LAYERS = {0: 'Layer I-III', 1: 'Layer IV', 2: 'Layer V-VI'}
    BOUNDARIES = list(zip(list(LAYERS.keys())[:-1], list(LAYERS.keys())[1:]))

    @classmethod
    def decode_meta(cls, spec):
        decoder_spec = spec.get("decoder", {})
        if decoder_spec["@type"].endswith('csv'):
            kwargs = {k: v for k, v in decoder_spec.items() if k != "@type"}
            return lambda b: pd.read_csv(BytesIO(b), **kwargs)
        else:
            return None


    class CorticalProfileConnector(ZipfileConnector):
        class ZipFileLoader:
            """
            Loads a file from the zip archive, but mimics the behaviour
            of cached http requests used in other connectors.
            """
            def __init__(self, zipfile, filename, decode_func, meta=None):
                self.zipfile = zipfile
                self.filename = filename
                self.func = decode_func
                self.cachefile = CACHE.build_filename(zipfile + filename)
                self.meta = meta

            @property
            def cached(self):
                return os.path.isfile(self.cachefile)

            @property
            def data(self):
                container = ZipFile(self.zipfile)
                df = self.func(container.open(self.filename).read())
                if self.meta is not None:
                    df = DFWithMeta(df)
                    df.meta = self.meta
                return df


        def get_loader(self, filename, folder="", decode_func=None):
            """Get a lazy loader for a file, for loading data
            only once loader.data is accessed."""
            meta = self.ZipFileLoader(self.zipfile, 'meta.json', lambda b: json.loads(b)).data
            loader = self.ZipFileLoader(self.zipfile, filename, decode_func, meta)
            return loader


    def __init__(
        self,
        url: str,
        anchor: _anchor.AnatomicalAnchor,
        datasets: list = [],
        prerelease: bool = False,
        id: str = None,
        region: Union[str, _region.Region] = None,
        connector: RepositoryConnector = None,
        decode_func: Callable = None

    ):
        """
        Generate a cell density profile from a URL to a cloud folder
        formatted according to the structure used by Bludau/Dickscheid et al.
        """
        id_template = re.compile(r'marmoset-nencki-monash-template--calbindin-profile-(?P<region_code>.*)')
        cortical_profile.CorticalProfile.__init__(
            self,
            description=self.DESCRIPTION,
            modality="Segmented cell body density",
            unit="detected cells / 0.1mm3",
            anchor=anchor,
            datasets=datasets,
            prerelease=prerelease,
            id=id,
        )
        matches = id_template.match(self._id)
        if matches:
            code = matches.group('region_code')
        else:
            raise ValueError('Malformed feature id %s' % (self.id,))
        self.region = region
        self.region_code = code
        self.file = f'{code}.csv'
        self._step = 0.01
        self._url = url
        self._connector = connector
        self._decode_func = decode_func

    @property
    def boundary_positions(self):
        """
        u = urlparse(self._url)
        pattern = u.path.rsplit('/', 1)[-1]
        url = self._url.replace(pattern, 'boundaries.json')
        #poly = self.poly_srt(np.array(requests.HttpRequest(url).get()["segments"]))
        data = requests.HttpRequest(url).get()
        return data
        """
        if self._boundary_positions is None:
            self._boundary_positions = {}
            for b in self.BOUNDARIES:
                XY = self.boundary_annotation(b).astype("int")
                self._boundary_positions[b] = self.depth_image[
                    XY[:, 1], XY[:, 0]
                ].mean()
        return self._boundary_positions

    @property
    def layers(self):
        return self._layer_loader.get()

    @property
    def _depths(self):
        return [d + self._step / 2 for d in np.arange(0, 1, self._step)]

    @property
    def key(self):
        assert len(self.species) == 1
        return create_key("{}_{}_{}".format(
            self.id,
            self.species[0]['name'],
            self.region_code
        ))

    @property
    def data(self):
        """
        Returns a matrix as a pandas dataframe.
        -------
        pd.DataFrame
            A square matrix with region names as the column and row names.
        """
        #parcellations = self.anchor.represented_parcellations()
        #logger.info('anchor %s', self.anchor)
        #logger.info('rep %s', self.anchor.represented_parcellations())
        #assert len(parcellations) == 1
        #parc = next(iter(parcellations))
        #logger.info('parcellation is %s and parc is %s', parcellations, parc)
        """
        with QUIET:
            indexmap = {
                i: parc.get_region(regionname, allow_tuple=True)
                for i, regionname in enumerate(self.regions)
            }
        """
        try:
            df = self._connector.get(self.file, decode_func=self._decode_func)
        except Exception:
            raise RuntimeError("Could not decode connectivity matrix regions.")
        return df.copy()

    def plot(self, *args, backend="matplotlib", **kwargs):
        """
        Plot the profile.

        Parameters
        ----------
        backend: str
            "matplotlib", "plotly", or others supported by pandas DataFrame
            plotting backend.
        **kwargs
            Keyword arguments are passed on to the plot command.
            'layercolor' can be used to specify a color for cortical layer shading.
        """
        wrapwidth = kwargs.pop("textwrap") if "textwrap" in kwargs else 40
        kwargs["title"] = kwargs.get("title", "\n".join(wrap(self.name, wrapwidth)))

        if backend == "matplotlib":
            raise NotImplemented('matplotlib graph is not supported yet')
        elif backend == "plotly":
            try:
                kwargs["title"] = kwargs["title"].replace("\n", "<br>")
                kwargs["labels"] = {
                    "index": kwargs.pop("xlabel", None) or kwargs.pop("index", "Cortical depth"),
                    "value": kwargs.pop("ylabel", None) or kwargs.pop("value", self._unit)
                }
                #fig = self.data.plot(*args, **kwargs, backend=backend)
                data = self.data
                cases = list(data.case_id.unique())
                fig = make_subplots(rows=1, cols=3, subplot_titles=cases)
                bp = data.meta['value']['layer_breakpoint'][self.region_code]
                for idx, c in enumerate(cases):
                    d = data[data.case_id==c]
                    fig.add_trace(go.Scatter(x=d.density_5, y=d.depth, line_color='#555555', mode='lines', name=c), row=1, col=(idx+1))
                    fig.add_trace(go.Scatter(x=d.density_25, y=d.depth, fill='tonexty', line_color='#555555', fillcolor='#eeeeee', mode='lines', name=c), row=1, col=(idx+1))
                    fig.add_trace(go.Scatter(x=d.density_50, y=d.depth, fill='tonexty', line_color='#000000', fillcolor='#cccccc', mode='lines', name=c), row=1, col=(idx+1))
                    fig.add_trace(go.Scatter(x=d.density_75, y=d.depth, fill='tonexty', line_color='#555555', fillcolor='#cccccc', mode='lines', name=c), row=1, col=(idx+1))
                    fig.add_trace(go.Scatter(x=d.density_95, y=d.depth, fill='tonexty', line_color='#555555', fillcolor='#eeeeee', mode='lines', name=c), row=1, col=(idx+1))
                    #fig.add_trace(go.Scatter(x=data.density_50, y=data.depth, mode='lines'), row=1, col=2)
                    #fig.add_trace(go.Scatter(x=data.density_50, y=data.depth, mode='lines'), row=1, col=3)

                    bp1 = bp['depth_1_3'][c]
                    bp2 = bp['depth_5_6'][c]
                    fig.add_hrect(
                        y0=0, y1=bp1, line_width=0, fillcolor="gray",
                        opacity=0.,
                        layer='below',
                        row=1, col=(idx+1)
                    )
                    fig.add_hrect(
                        y0=bp1, y1=bp2, line_width=0, fillcolor="gray",
                        opacity=0.2,
                        layer='below',
                        row=1, col=(idx+1)
                    )
                    fig.add_hrect(
                        y0=bp2, y1=1, line_width=0, fillcolor="gray",
                        opacity=0.,
                        layer='below',
                        row=1, col=(idx+1)
                    )
                    if idx == len(cases) - 1:
                        fig.add_hrect(
                            y0=0, y1=bp1, line_width=0, fillcolor="gray",
                            opacity=0.,
                            label=dict(text='<b>I-III</b>', textposition="middle right", font={'family': 'Arial'}),
                            row=1, col=3
                        )
                        fig.add_hrect(
                            y0=bp1, y1=bp2, line_width=0, fillcolor="gray",
                            opacity=0.,
                            label=dict(text='<b>VI</b>', textposition="middle right", font={'family': 'Arial'}),
                            row=1, col=3
                        )
                        fig.add_hrect(
                            y0=bp2, y1=1, line_width=0, fillcolor="gray",
                            opacity=0.,
                            label=dict(text='<b>V-VI</b>', textposition="middle right", font={'family': 'Arial'}),
                            row=1, col=3
                        )
                fig.update_yaxes(autorange='reversed', title='normalized cortical depth', row=1, col=1)
                fig.update_yaxes(autorange='reversed', showticklabels=False, row=1, col=2)
                fig.update_yaxes(autorange='reversed', showticklabels=False, row=1, col=3)
                max_x = math.ceil(data.density_95.max())
                fig.update_xaxes(title='CB+ density<br>(·10<sup>3</sup> ·mm<sup>-3</sup>)', range=[0, max_x], row=1, col=1)
                fig.update_xaxes(title='CB+ density<br>(·10<sup>3</sup> ·mm<sup>-3</sup>)', range=[0, max_x], row=1, col=2)
                fig.update_xaxes(title='CB+ density<br>(·10<sup>3</sup> ·mm<sup>-3</sup>)', range=[0, max_x], row=1, col=3)
                fig.update_annotations(font_family='Arial')

                fig.update_layout(
                    showlegend=False,
                    font=dict(
                        family='Arial'
                    ),
                    xaxis=dict(
                        tickfont=dict(family='Arial')
                    ),
                    yaxis=dict(
                        tickfont=dict(family='Arial')
                    ),
                    #yaxis_range=(0, max(data)),
                    title=dict(
                        #automargin=True,
                        yref="container",
                        xref="container",
                        pad=dict(t=40), xanchor="left", yanchor="top",

                    )
                )
            except Exception:
                logger.info('error', exc_info=True)
            return fig
        else:
            return self.data.plot(*args, **kwargs, backend=backend)


