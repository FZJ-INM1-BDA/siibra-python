# Copyright 2018-2025
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

from typing import List, Union
import sqlite3
from io import BytesIO

import pandas as pd

from . import query as _query
from ..retrieval import requests
from ..features import tabular, anchor as _anchor
from ..core import region
from ..locations import BoundingBox, Point, PointSet

DATABASE_URL = "https://data-proxy.ebrains.eu/api/v1/buckets/d-41673110-f3eb-43cd-9d9c-c845c6f0573c/lfp_atlas_bids.sqlite"

NAMED_ARGS = ["pathology", "pharmacology", "signal_quality"]


@staticmethod
def database_as_df() -> pd.DataFrame:
    req = requests.HttpRequest(
        DATABASE_URL,
        msg_if_not_cached="Downloading sql database for LSF query.",
    )
    req._retrieve()
    db = sqlite3.connect(req.cachefile)
    return pd.read_sql("SELECT * FROM bids", db)


@staticmethod
def get_arg_options():
    df = database_as_df()
    return {col: set(df[col]) for col in NAMED_ARGS}


class LFPQuery(
    _query.LiveQuery,
    args=NAMED_ARGS,
    FeatureType=tabular.LocalFieldPotential,
):

    def resolve_db_rows(self, concept: Union[region.Region, BoundingBox]) -> pd.DataFrame:
        df = database_as_df()
        if isinstance(concept, region.Region):
            unique_regions = set(df["whs_label"].unique())
            if concept.name in unique_regions:
                mask = df["whs_label"] == concept.name
            else:
                regions = [
                    d.name for d in concept.descendants if d.name in unique_regions
                ]
                mask = df["whs_label"].isin(regions)
        elif isinstance(concept, BoundingBox):
            entries_as_ptcld = PointSet(
                df[["whs_x", "whs_y", "whs_z"]],
                space="minds/core/referencespace/v1.0.0/d5717c4a-0fa1-46e6-918c-b8003069ade8",
            )
            intersecting_points = entries_as_ptcld.intersection(
                concept
            )  # returns a reduced PointSet with og indices as labels if the concept is a region
            mask = pd.Series([False] * len(df), index=df.index)
            mask.iloc[intersecting_points.labels] = True
        else:
            raise ValueError(f"{concept} is neither Region nor Location")
        # Apply filters for non-None kwargs
        if self.pathology is not None:
            mask &= df["pathology"] == self.pathology
        if self.pharmacology is not None:
            mask &= df["pharmacology"] == self.pharmacology
        if self.signal_quality is not None:
            mask &= df["signal_quality"] == self.signal_quality
        return df[mask]

    def __init__(self, **kwargs):
        self.pathology = kwargs["pathology"] = kwargs.get("pathology", None)
        self.signal_quality = kwargs["signal_quality"] = kwargs.get(
            "signal_quality", None
        )
        self.pharmacology = kwargs["pharmacology"] = kwargs.get("pharmacology", None)
        _query.LiveQuery.__init__(self, **kwargs)

    def query(
        self, concept: Union[region.Region, BoundingBox]
    ) -> List[tabular.LocalFieldPotential]:
        df = self.resolve_db_rows(concept)
        return [
            tabular.LocalFieldPotential(
                db_entry=row,
                anchor=_anchor.AnatomicalAnchor(
                    location=Point([row.whs_x, row.whs_y, row.whs_z], "waxholm"),
                    region=row.whs_label if row.whs_label != "Clear Label" else None,
                    species="RATTUS NORVEGICUS",
                ),
            )
            for row in df.itertuples()
        ]


class RegionalLFPQuery(
    LFPQuery,
    args=NAMED_ARGS,
    FeatureType=tabular.RegionalLocalFieldPotential,
):

    BUCKET_URL = "https://data-proxy.ebrains.eu/api/v1/buckets/regional-local-field-potentials-rat/RegionalLocalFieldPotential_v1/PowerSpectrums/"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_options(cls, df: pd.DataFrame = None):
        df = cls.database_as_df() if df is None else df
        options = {
            whs_label: {
                (row.pathology, row.pharmacology, row.signal_quality)
                for row in df[df["whs_label"] == whs_label].itertuples(index=False)
            }
            for whs_label in df["whs_label"].unique()
        }
        return options

    @classmethod
    def create_loader(cls, regionname, pat, phar, sq):
        filename = f"group-{regionname.lower().replace(',', '')}"
        filename += f"_pathology-{pat}_pharmacology-{phar}_signal_quality-{sq}"
        filename = filename.replace(' ', '_')
        filename += "_LocalFieldPotential.tsv"
        return requests.HttpRequest(
            cls.BUCKET_URL + filename,
            func=lambda b: pd.read_csv(BytesIO(b), sep=",", index_col=0)
        )

    def query(self, concept):
        df = self.resolve_db_rows(concept)
        results = []
        available_tuples = {
            (row.pathology, row.pharmacology, row.signal_quality)
            for row in df.itertuples()
        }
        for whs_label in set(df["whs_label"]):
            for pat, phar, sq in available_tuples:
                mask = (
                    (df["whs_label"] == whs_label)
                    & (df["pathology"] == pat)
                    & (df["pharmacology"] == phar)
                    & (df["signal_quality"] == sq)
                )
                if not mask.any():
                    continue
                anchor = _anchor.AnatomicalAnchor(
                    location=PointSet(
                        df[mask][["whs_x", "whs_y", "whs_z"]].values,
                        space="waxholm"
                    ),
                    region=whs_label,
                    species="RATTUS NORVEGICUS",
                )
                results.append(
                    tabular.RegionalLocalFieldPotential(
                        anchor=anchor,
                        db_entries=df[mask],
                        pathology=pat,
                        pharmacology=phar,
                        signal_quality=sq,
                        loader=self.create_loader(whs_label, pat, phar, sq)
                    )
                )
        return results
