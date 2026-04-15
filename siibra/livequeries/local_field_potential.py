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

from typing import List
import sqlite3

import pandas as pd

from . import query as _query
from ..retrieval import requests
from ..features import tabular, anchor as _anchor
from ..core import structure, region
from ..locations import Location, Point, PointCloud


class LFPQuery(
    _query.LiveQuery,
    args=["pathology", "pharmacology", "signal_quality"],
    FeatureType=tabular.LocalFieldPotential,
):
    DATABASE_URL = "https://data-proxy.ebrains.eu/api/v1/buckets/d-41673110-f3eb-43cd-9d9c-c845c6f0573c/lfp_atlas_bids.sqlite"

    @classmethod
    def database_as_df(cls) -> pd.DataFrame:
        req = requests.HttpRequest(
            cls.DATABASE_URL,
            msg_if_not_cached="Downloading sql database for LSF query.",
        )
        req._retrieve()
        db = sqlite3.connect(req.cachefile)
        return pd.read_sql("SELECT * FROM bids", db)

    def resolve_db_rows(
        self,
        concept: structure.BrainStructure
    ) -> pd.DataFrame:
        df = self.database_as_df()
        if isinstance(concept, region.Region):
            unique_regions = set(df["whs_label"].unique())
            if concept.name in unique_regions:
                mask = df["whs_label"] == concept.name
            else:
                regions = [d for d in concept.descendants if d.name in unique_regions]
                mask = df["whs_label"].isin(regions)
        elif isinstance(concept, Location):
            entries_as_ptcld = PointCloud(
                df[["whs_x", "whs_y", "whs_z"]],
                space="minds/core/referencespace/v1.0.0/d5717c4a-0fa1-46e6-918c-b8003069ade8",
            )
            intersecting_points = entries_as_ptcld.intersection(
                concept
            )  # returns a reduced PointCloud with og indices as labels if the concept is a region
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
        _query.LiveQuery.__init__(self, **kwargs)
        self.pathology = kwargs.get("pathology", None)
        self.signal_quality = kwargs.get("signal_quality", None)
        self.pharmacology = kwargs.get("pharmacology", None)

    def query(self, concept: structure.BrainStructure) -> List[tabular.LocalFieldPotential]:
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


class FixedLFPQuery(
    LFPQuery,
    args=["pathology", "pharmacology", "signal_quality"],
    FeatureType=tabular.LocalFieldPotentialSpectrum,
):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def query(self, concept):
        df = self.resolve_db_rows(concept)
        fixed_tpls = {
            whs_label: {
                (row.pathology, row.pharmacology, row.signal_quality)
                for row in df[df["whs_label"] == whs_label].itertuples(index=False)
            }
            for whs_label in df["whs_label"].unique()
        }
        results = []
        for whs_label, options in fixed_tpls.items():
            for pat, phar, sq in options:
                mask = df["whs_label"] == whs_label
                mask &= df["pathology"] == pat
                mask &= df["pharmacology"] == phar
                mask &= df["signal_quality"] == sq
                anchor = _anchor.AnatomicalAnchor(
                    location=PointCloud(df[["whs_x", "whs_y", "whs_z"]].values, "waxholm"),
                    region=whs_label,
                    species="RATTUS NORVEGICUS",
                )
                results.append(
                    tabular.LocalFieldPotentialSpectrum(
                        anchor=anchor,
                        db_entries=df[mask],
                        pathology=pat,
                        pharmacology=phar,
                        signal_quality=sq,
                    )
                )
        return results
