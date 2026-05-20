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

from typing import TYPE_CHECKING, Dict, Literal, List, Union
from hashlib import md5
import numpy as np
import pandas as pd

from . import tabular
from ...retrieval.requests import HttpRequest
from ...commons import logger, siibra_tqdm
from ...exceptions import MissingFileException

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
    def name(self):
        return (
            super().name + " - "
            + str(
                {
                    attr: getattr(self, attr)
                    for attr in [
                        "pathology",
                        "pharmacology",
                        "signal_quality",
                    ]
                }
            )[1:-1]
        )

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
        if self._db_entry.lfp_file is np.nan:
            raise MissingFileException(f"{self} does not have a lfp file.")
        return HttpRequest(BASE_URL.format(filepath=self._db_entry.lfp_file)).get()

    def get_psd_file(self):
        if self._db_entry.psd_file is np.nan:
            raise MissingFileException(f"{self} does not have a psd file.")
        return HttpRequest(BASE_URL.format(filepath=self._db_entry.psd_file)).get()

    def get_motion_file(self):
        if self._db_entry.motion_file is np.nan:
            raise MissingFileException(f"{self} does not have a motion file.")
        url = BASE_URL.format(filepath=self._db_entry.motion_file)
        return HttpRequest(url).get()

    @classmethod
    def get_spectrum(
        cls,
        lfps: List["LocalFieldPotential"],
        spectrum_type: Literal["spectrogram", "spectrogram_rhythmic", "spectrogram_arrhythmic"],
    ):
        return cls._get_spectrum(
            pd.DataFrame([lfp._db_entry for lfp in lfps]),
            spectrum_type=spectrum_type,
        )

    @classmethod
    def _get_spectrum(
        cls,
        lfp_entries: pd.DataFrame,
        spectrum_type: Literal["spectrogram", "spectrogram_rhythmic", "spectrogram_arrhythmic"],
    ):
        def get_psd_file(row):
            return HttpRequest(BASE_URL.format(filepath=row.psd_file)).get()

        n_files = len(lfp_entries)
        for first_succes, row in enumerate(lfp_entries.itertuples()):
            if row.psd_file is np.nan:
                logger.info(f"{row} is missing the psd file. Excluding it from spectrum.")
                n_files -= 1
                continue
            with get_psd_file(row) as psdf:
                times = psdf["/times"][:]
                freqs = psdf["/frequencies"][:]
                spectrogram = psdf[f"/{spectrum_type}"][:]
                P_m = np.nanmedian(spectrogram, axis=0)  # Average over time
            times = times.flatten()
            freqs = freqs.flatten()
            break

        # Load the rest and find the median
        for row in siibra_tqdm(
            lfp_entries.iloc[first_succes + 1:].itertuples(),
            total=len(lfp_entries.iloc[first_succes + 1:]),
            unit="entry",
        ):
            if row.psd_file is np.nan:
                logger.info(f"{row} is missing the psd file. Excluding it from spectrum.")
                n_files -= 1
                continue
            with get_psd_file(row) as psdf:
                P_m = P_m + np.nanmedian(psdf[f"/{spectrum_type}"][:], axis=0)

        # Calculate average
        P_m = P_m / n_files
        return pd.DataFrame(P_m, index=freqs)

    @classmethod
    def plot_spectrum(
        cls,
        lfps: List["LocalFieldPotential"],
        spectrum_type: Literal[
            "spectrogram", "spectrogram_rhythmic", "spectrogram_arrhythmic"
        ] = "spectrogram_rhythmic",
        backend="matplotlib",
        **kwargs,
    ):
        df = cls.get_spectrum(lfps=lfps, spectrum_type=spectrum_type)
        match spectrum_type:
            case "spectrogram":
                ylabel = "dB"
            case "spectrogram_rhythmic":
                ylabel = " dB(fractal)"
            case "spectrogram_arrhythmic":
                ylabel = "dB"
        kwargs["xlabel"] = kwargs.get("xlabel", "Frequency (Hz)")
        kwargs["ylabel"] = kwargs.get("xlabel", ylabel)
        kwargs["label"] = "smoothed median"
        return df.plot(backend=backend, **kwargs)


class LocalFieldPotentialSpectrum(tabular.Tabular, category="functional"):

    DESCRIPTION = """"""
    ID_TEMPLATE = "41673110-f3eb-43cd-9d9c-c845c6f0573c--{indices_as_hex}"

    def __init__(
        self,
        anchor: "AnatomicalAnchor",
        db_entries: pd.DataFrame,
        spectrum_type: Literal[
            "spectrogram", "spectrogram_rhythmic", "spectrogram_arrhythmic"
        ],
        pathology: Union[str, None] = None,
        pharmacology: Union[str, None] = None,
        signal_quality: Union[str, None] = None,
    ):
        tabular.Tabular.__init__(
            self,
            description=self.DESCRIPTION,
            modality=f"Local field potential spectrum - {spectrum_type}",
            anchor=anchor,
            id=self.ID_TEMPLATE.format(
                indices_as_hex=md5(str(db_entries.index).encode("utf-8")).hexdigest()
            ),
        )
        self._db_entries = db_entries
        self.pharmacology = pharmacology
        self.pathology = pathology
        self.signal_quality = signal_quality
        self.spectrum_type = spectrum_type

    @property
    def name(self):
        return (
            super().name + " - "
            + str(
                {
                    attr: getattr(self, attr)
                    for attr in [
                        "pathology",
                        "pharmacology",
                        "signal_quality",
                        "spectrum_type",
                    ]
                }
            )[1:-1]
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
        return LocalFieldPotential._get_spectrum(
            lfp_entries=self._db_entries,
            spectrum_type=self.spectrum_type,
        )

    def plot(self, *args, backend="matplotlib", **kwargs):
        kwargs["kind"] = "line"
        match self.spectrum_type:
            case "spectrogram":
                ylabel = "dB"
            case "spectrogram_rhythmic":
                ylabel = " dB(fractal)"
            case "spectrogram_arrhythmic":
                ylabel = "dB"
        kwargs["xlabel"] = kwargs.get("xlabel", "Frequency (Hz)")
        kwargs["ylabel"] = kwargs.get("xlabel", ylabel)
        kwargs["label"] = "smoothed median"
        return super().plot(*args, backend=backend, **kwargs)
