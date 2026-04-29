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

from typing import TYPE_CHECKING, Dict, Literal
from hashlib import md5
import numpy as np
from pandas import DataFrame

from . import tabular
from ...retrieval.requests import HttpRequest
from ...commons import logger, siibra_tqdm

if TYPE_CHECKING:
    from ..anchor import AnatomicalAnchor


BASE_URL = "https://data-proxy.ebrains.eu/api/v1/buckets/d-41673110-f3eb-43cd-9d9c-c845c6f0573c/{filepath}"


class LocalFieldPotential(tabular.Tabular, category="functional"):

    DESCRIPTION = """"""
    ID_TEMPLATE = "41673110-f3eb-43cd-9d9c-c845c6f0573c--{index}"

    def __init__(
        self,
        anchor: "AnatomicalAnchor",
        db_entry: Dict,
    ):
        tabular.Tabular.__init__(
            self,
            description=self.DESCRIPTION,
            modality="Local field potential",
            anchor=anchor,
            id=self.ID_TEMPLATE.format(index=db_entry.Index),
        )
        self._db_entry = db_entry

    @property
    def pharmacology(self):
        return self._db_entry.pharmacology

    @property
    def pathology(self):
        return self._db_entry.pathology

    @property
    def signal_quality(self):
        return self._db_entry.signal_quality

    @property
    def subject(self):
        return self._db_entry.subject

    @property
    def session(self):
        return self._db_entry.session

    @property
    def coordinates(self):
        return self.anchor.location

    def get_lfp_file(self):
        return HttpRequest(BASE_URL.format(self._db_entry["lfp_file"])).get()

    def get_psd_file(self):
        return HttpRequest(BASE_URL.format(self._db_entry["psd_file"])).get()

    def get_motion_file(self):
        return HttpRequest(BASE_URL.format(self._db_entry["motion_file"])).get()


class LocalFieldPotentialSpectrum(tabular.Tabular, category="functional"):
    DESCRIPTION = """"""
    ID_TEMPLATE = "41673110-f3eb-43cd-9d9c-c845c6f0573c--{indices_as_hex}"

    def __init__(
        self,
        anchor: "AnatomicalAnchor",
        db_entries: DataFrame,
        pathology: Literal[
            "lesioned hemisphere in 6-OHDA hemilesioned animal",
            "intact hemisphere in 6-OHDA hemilesioned animal",
            "none",
        ],
        pharmacology: Literal[
            "baseline",
            "levodopa",
            "mk801",
            "ketamine",
            "lsd",
            "amphetamine",
            "doi",
            "pcp",
            "sumanirole",
            "skf82958",
        ],
        signal_quality: Literal["typical", "atypical", "strongly atypical"],
    ):
        tabular.Tabular.__init__(
            self,
            description=self.DESCRIPTION,
            modality="Local field potential spectrum",
            anchor=anchor,
            id=self.ID_TEMPLATE.format(
                indices_as_hex=md5(str(db_entries.index).encode("utf-8")).hexdigest()
            ),
        )
        self._db_entries = db_entries
        self.pharmacology = pharmacology
        self.pathology = pathology
        self.signal_quality = signal_quality

    @classmethod
    def get_options(cls):
        from ...livequeries.local_field_potential import FixedLFPQuery

        return FixedLFPQuery().get_options()

    @property
    def name(self):
        return (
            super().name
            + f"pathology: {self.pathology}, pharmacology: {self.pharmacology}, signal_quality: {self.signal_quality}"
        )

    @property
    def subjects(self):
        return set(self._db_entries["subject"].unique())

    @property
    def session(self):
        return self._db_entries["session"]

    @property
    def coordinates(self):
        return self.anchor.location

    def get_psd_file(self, index: int):
        return HttpRequest(
            BASE_URL.format(filepath=self._db_entries.loc[index, :]["psd_file"])
        ).get()

    @property
    def data(self):
        nfiles = self._db_entries["psd_file"].unique()
        logger.info("Loading first file")
        index = self._db_entries.index[0]
        with self.get_psd_file(index) as f:
            times = f["/times"][:]
            freqs = f["/frequencies"][:]
            spectrogram = f["/spectrogram_rhythmic"][:]
            P_m = np.nanmedian(spectrogram, axis=0)  # Average over time
        times = times.flatten()
        freqs = freqs.flatten()

        # Load the rest and find the median
        for index, row in siibra_tqdm(self._db_entries.iloc[1:].iterrows()):
            with self.get_psd_file(index) as f:
                P_m = P_m + np.nanmedian(f["/spectrogram_rhythmic"][:], axis=0)

        # Calculate average
        P_m = P_m / nfiles
        logger.info("\nDone loading.")
        return DataFrame(P_m, index=freqs)

    def plot(self, *args, backend="matplotlib", **kwargs):
        kwargs["kind"] = "line"
        kwargs["xlabel"] = kwargs.get("xlabel", "Frequency (Hz)")
        kwargs["ylabel"] = kwargs.get("xlabel", "dB(fractal)")
        return super().plot(*args, backend=backend, **kwargs)
