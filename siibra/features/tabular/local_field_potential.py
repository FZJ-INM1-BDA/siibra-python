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

from typing import TYPE_CHECKING, Dict, List, Union
from hashlib import md5
import numpy as np
import pandas as pd

from . import tabular
from ...commons import logger, siibra_tqdm
from ...retrieval import SiibraHttpRequestError
from ...retrieval.requests import HttpRequest
from ...retrieval.datasets import EbrainsV3DatasetVersion

if TYPE_CHECKING:
    from ..anchor import AnatomicalAnchor


class MissingFileException(Exception):
    pass


BASE_URL = "https://data-proxy.ebrains.eu/api/v1/buckets/d-41673110-f3eb-43cd-9d9c-c845c6f0573c/{filepath}"


@staticmethod
def _get_spectra(lfp_entries: pd.DataFrame):
    """
    Compute an average spectrum from local field potential metadata entries.

    For each entry with an available power spectral density file, the selected
    spectrogram dataset is loaded, summarized across time by the median, and
    accumulated across entries. Entries without a power spectral density file
    are skipped.

    Parameters
    ----------
    lfp_entries : pandas.DataFrame
        Metadata entries describing local field potential recordings. The
        table must contain a ``psd_file`` column.

    Returns
    -------
    pandas.DataFrame
        Average spectrum indexed by frequency.

    Notes
    -----
    This method downloads the required power spectral density files while
    computing the result.
    """
    def get_psd_file(row):
        return HttpRequest(BASE_URL.format(filepath=row.psd_file)).get()

    n_files = len(lfp_entries)
    spectrum_types = ["spectrogram", "spectrogram_rhythmic", "spectrogram_arrhythmic"]
    P_m = {}
    for first_succes, row in enumerate(lfp_entries.itertuples()):
        if row.psd_file is np.nan:
            logger.info(f"{row} is missing the psd file. Excluding it from spectrum.")
            n_files -= 1
            continue
        with get_psd_file(row) as psdf:
            times = psdf["/times"][:]
            freqs = psdf["/frequencies"][:]
            for st in spectrum_types:
                P_m[st] = np.nanmedian(psdf[f"/{st}"][:], axis=0)  # Average over time
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
        try:
            with get_psd_file(row) as psdf:
                for st in spectrum_types:
                    P_m[st] = P_m[st] + np.nanmedian(psdf[f"/{st}"][:], axis=0)
        except SiibraHttpRequestError:
            logger.error(f"Broken url: {BASE_URL.format(filepath=row.psd_file)}")
            n_files -= 1
            continue

    spectra = pd.DataFrame(P_m, index=freqs)
    spectra /= n_files  # Calculate average
    spectra.rename(
        columns={
            "spectrogram": "spectrogram (dB)",
            "spectrogram_rhythmic": "spectrogram rhythmic (dB (fractal))",
            "spectrogram_arrhythmic": "spectrogram arrhythmic (dB)",
        },
        inplace=True
    )
    return spectra


class LocalFieldPotential(tabular.Tabular, category="functional"):
    """
    Local field potential recording anchored to Waxholm coordinates and potentially a region.
    """

    ID_TEMPLATE = "41673110-f3eb-43cd-9d9c-c845c6f0573c--{index}"

    def __init__(
        self,
        anchor: "AnatomicalAnchor",
        db_entry: Dict,
    ):
        """
        Initialize a local field potential feature.

        Parameters
        ----------
        anchor : AnatomicalAnchor
            Anatomical location associated with the recording.
        db_entry : dict
            Metadata entry describing the recording and its associated files.
        """
        tabular.Tabular.__init__(
            self,
            description=None,
            modality="Local field potential",
            anchor=anchor,
            id=self.ID_TEMPLATE.format(index=db_entry.Index),
            datasets=[EbrainsV3DatasetVersion("41673110-f3eb-43cd-9d9c-c845c6f0573c")],
            data=None
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
        """
        Fetch the local field potential data file.

        Returns
        -------
        h5-file-like
            Open file handle returned by the HTTP request.

        Raises
        ------
        MissingFileException
            If no local field potential file is available for this entry.
        """
        if self._db_entry.lfp_file is np.nan:
            raise MissingFileException(f"{self} does not have a lfp file.")
        return HttpRequest(BASE_URL.format(filepath=self._db_entry.lfp_file)).get()

    def get_psd_file(self):
        """
        Fetch the power spectral density file.

        Returns
        -------
        h5-file-like
            Open file handle returned by the HTTP request.

        Raises
        ------
        MissingFileException
            If no power spectral density file is available for this entry.
        """
        if self._db_entry.psd_file is np.nan:
            raise MissingFileException(f"{self} does not have a psd file.")
        return HttpRequest(BASE_URL.format(filepath=self._db_entry.psd_file)).get()

    def get_motion_file(self):
        """
        Fetch the motion data file.

        Returns
        -------
        h5-file-like
            Open file handle returned by the HTTP request.

        Raises
        ------
        MissingFileException
            If no motion file is available for this entry.
        """
        if self._db_entry.motion_file is np.nan:
            raise MissingFileException(f"{self} does not have a motion file.")
        url = BASE_URL.format(filepath=self._db_entry.motion_file)
        return HttpRequest(url).get()

    @classmethod
    def get_spectra(cls, lfps: List["LocalFieldPotential"]):
        """
        Compute an average spectrum from local field potential features.

        Parameters
        ----------
        lfps : list of LocalFieldPotential
            Local field potential features to include in the spectrum.

        Returns
        -------
        pandas.DataFrame
            Average spectrum indexed by frequency.
        """
        return _get_spectra(pd.DataFrame((lfp._db_entry for lfp in lfps)))

    @classmethod
    def plot_spectra(
        cls,
        lfps: List["LocalFieldPotential"],
        backend="matplotlib",
        **kwargs,
    ):
        """
        Plot an average spectrum from local field potential features.

        Parameters
        ----------
        lfps : list of LocalFieldPotential
            Local field potential features to include in the spectrum.
        backend : str, optional
            Plotting backend passed to :meth:`pandas.DataFrame.plot`.
            The default is ``"matplotlib"``.
        **kwargs
            Additional keyword arguments passed to :meth:`pandas.DataFrame.plot`.

        Returns
        -------
        object
            Plot object returned by the selected pandas plotting backend.
        """
        data = cls.get_spectra(lfps=lfps)
        if backend == "matplotlib":
            kwargs["xlabel"] = kwargs.get("xlabel", "Frequency (Hz)")
            kwargs["grid"] = "major"
            kwargs["subplots"] = True
            data.plot(backend=backend, **kwargs)
        elif backend == "plotly":
            kwargs["facet_row"] = 'variable'
            fig = data.plot(backend=backend, **kwargs)
            facet_labels = [
                ann.text.replace("variable=", "")
                for ann in fig.layout.annotations
            ]
            fig.update_yaxes(matches=None)
            fig.update_layout(showlegend=False)
            fig.for_each_annotation(lambda a: a.update(text=""))
            for axis, label in zip(
                [fig.layout[f"yaxis{i if i > 1 else ''}"] for i in range(1, len(facet_labels) + 1)],
                facet_labels[::-1],
            ):
                axis.title.text = label
            fig.update_layout(height=900)
            fig.layout.xaxis.title.text = "Frequency (Hz)"
            return fig
        else:
            return data.plot(backend=backend, **kwargs)


class RegionalLocalFieldPotential(tabular.Tabular, category="functional"):
    """
    Local field potential recording anchored to Waxholm region and a preselected spectrum type.
    """
    ID_TEMPLATE = "41673110-f3eb-43cd-9d9c-c845c6f0573c--{indices_as_hex}"

    def __init__(
        self,
        anchor: "AnatomicalAnchor",
        db_entries: pd.DataFrame,
        pathology: Union[str, None] = None,
        pharmacology: Union[str, None] = None,
        signal_quality: Union[str, None] = None,
    ):
        tabular.Tabular.__init__(
            self,
            description=None,
            data=None,
            modality="Regional local field potential",
            anchor=anchor,
            id=self.ID_TEMPLATE.format(
                indices_as_hex=md5(str(db_entries.index).encode("utf-8")).hexdigest()
            ),
            datasets=[EbrainsV3DatasetVersion("41673110-f3eb-43cd-9d9c-c845c6f0573c")],
        )
        self._db_entries = db_entries
        self.pharmacology = pharmacology
        self.pathology = pathology
        self.signal_quality = signal_quality

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
    def subjects(self):
        return set(self._db_entries["subject"].unique())

    @property
    def session(self):
        return self._db_entries["session"]

    @property
    def coordinates(self):
        return self.anchor.location

    @property
    def data(self):
        """
        Compute the smoothed median spectra represented by this feature.

        Returns
        -------
        pandas.DataFrame
            Average spectra by type and indexed by frequency.

        Notes
        -----
        Accessing this property downloads the required power spectral density
        files and recomputes the spectrum.
        """
        return _get_spectra(lfp_entries=self._db_entries)

    def plot(self, *args, backend="matplotlib", **kwargs):
        """
        Plot the average local field potential spectrum.

        Parameters
        ----------
        *args
            Positional arguments passed to the parent plotting method.
        backend : str, optional
            Plotting backend passed to the parent plotting method.
            The default is ``"matplotlib"``.
        **kwargs
            Additional keyword arguments passed to the parent plotting method.

        Returns
        -------
        object
            Plot object returned by the selected backend.
        """
        kwargs["title"] = self.name
        if backend == "matplotlib":
            kwargs["xlabel"] = kwargs.get("xlabel", "Frequency (Hz)")
            kwargs["grid"] = "major"
            kwargs["subplots"] = True
            self.data.plot(*args, backend=backend, **kwargs)
        elif backend == "plotly":
            kwargs["facet_row"] = 'variable'
            fig = self.data.plot(*args, backend=backend, **kwargs)
            facet_labels = [
                ann.text.replace("variable=", "")
                for ann in fig.layout.annotations
            ]
            fig.update_yaxes(matches=None)
            fig.update_layout(showlegend=False)
            fig.for_each_annotation(lambda a: a.update(text=""))
            for axis, label in zip(
                [fig.layout[f"yaxis{i if i > 1 else ''}"] for i in range(1, len(facet_labels) + 1)],
                facet_labels[::-1],
            ):
                axis.title.text = label
            fig.update_layout(height=900)
            fig.layout.xaxis.title.text = "Frequency (Hz)"
            return fig
        else:
            return self.data.plot(*args, backend=backend, **kwargs)
