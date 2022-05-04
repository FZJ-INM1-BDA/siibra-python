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

from .. import QUIET, __version__, logger
from ..core import Atlas, Parcellation, Space
from ..features import get_features, modalities

import matplotlib.pyplot as plt
import nibabel as nib
import os
from nilearn import plotting, image
from fpdf import FPDF
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from typing import Union

matplotlib.use("Agg")


class AnatomicalAssignment:

    def __init__(
        self,
        parcellation: Union[str, Parcellation] = "julich 2.9",
        space: Union[str, Space] = "mni152",
        maptype="continuous",
        min_correlation: float = 0.3,
        min_entries: int = 4,
        resolution_dpi: float = 300,
        max_conn: int = 30,
        force_overwrite=False
    ):

        parcobj = Parcellation.REGISTRY[parcellation]
        spaceobj = Space.REGISTRY[space]

        self.min_correlation = min_correlation
        self.min_entries = min_entries
        self.dpi = resolution_dpi
        self.max_conn = max_conn
        self.overwrite = force_overwrite

        atlas = Atlas.REGISTRY.MULTILEVEL_HUMAN_ATLAS
        self.pmaps = atlas.get_map(
            parcellation=parcobj, space=spaceobj, maptype=maptype
        )

        # TODO do not just fetch the first connectivity source - choose one explicitly
        self.conn = get_features(self.pmaps.parcellation, modalities.StreamlineCounts)[
            0
        ]

    def run(self, niftifile, outdir=None):

        if outdir is None:
            from tempfile import mkdtemp

            outdir = mkdtemp()

        plotdir = os.path.join(outdir, "plots")
        if not os.path.isdir(plotdir):
            os.makedirs(plotdir)

        reportfile = f"{os.path.join(outdir,os.path.basename(niftifile))}.pdf"
        if os.path.isfile(reportfile) and not self.overwrite:
            logger.warn(
                f"File {reportfile} exists - skipping analysis for {niftifile}."
            )
            return reportfile

        logger.info(
            f"Analyzing {niftifile}. Results will be stored in folder {outdir}."
        )
        img = nib.load(niftifile)
        input_plot = self._plot_input(img, niftifile, plotdir)

        # get initial assignments and detected components
        initial_assignments, compimg = self.pmaps.assign(img)
        initial_assignments.sort_values(by="Correlation", ascending=False, inplace=True)

        assignments = self._select_assignments(initial_assignments, compimg, img)

        # plot the complete component image
        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=self.dpi)
        plotting.plot_glass_brain(compimg, axes=ax, alpha=0.3, cmap="Set1")
        # fig.tight_layout(pad=0.0)
        plt.ioff()
        components_plot = (
            f"{os.path.join(plotdir, os.path.basename(niftifile))}.components.png"
        )
        fig.savefig(components_plot, dpi=self.dpi)

        # plot masks of individual components
        component_plots = {}
        arr = np.asanyarray(compimg.dataobj)
        for component in tqdm(
            assignments.component.unique(),
            desc="Plotting component masks...",
            unit="components",
        ):
            component_plots[component] = self._plot_component(
                arr, component, compimg.affine, niftifile, plotdir
            )

        # plot relevant probability maps
        pmap_plots = {}
        for regionname in tqdm(
            assignments.region.unique(),
            desc="Plotting probability maps...",
            unit="maps",
        ):
            pmap_plots[regionname] = self._plot_pmap(regionname, plotdir)

        # plot relevant connectivity profiles
        profile_plots = {}
        for regionname in tqdm(
            assignments.region.unique(),
            desc="Plotting connectivity profiles...",
            unit="profiles",
        ):
            profile_plots[regionname] = self._plot_profile(
                self.conn, regionname, plotdir
            )
        not_found = [k for k, v in profile_plots.items() if v is None]
        if not_found:
            logger.warning(
                "No profiles found in connectivity matrix for regions "
                f"{', '.join(not_found)}"
            )

        # build the actual pdf report
        self._build_pdf(
            assignments,
            niftifile,
            input_plot,
            components_plot,
            component_plots,
            pmap_plots,
            profile_plots,
            reportfile,
        )
        return reportfile

    def _select_assignments(self, initial_assignments, compimg, img):

        compimg_res = image.resample_to_img(compimg, img, interpolation="nearest")
        comparr = np.asanyarray(compimg_res.dataobj)
        results = []
        for component_id in range(1, initial_assignments.Component.max() + 1):

            # compute centroid of component in MNI space
            comp_mask = comparr == component_id
            max_val = np.max(img.get_fdata()[comp_mask])
            X, Y, Z = np.where((img.get_fdata() == max_val) & comp_mask)
            centroid = np.dot(compimg_res.affine, [X[0], Y[0], Z[0], 1])

            # select strong assignments for this component
            for count, (index, row) in enumerate(
                initial_assignments[lambda df: df.Component == component_id].iterrows()
            ):

                # region = self.pmaps.parcellation.decode_region(row.Region)
                if (count >= self.min_entries) & (
                    row.Correlation < self.min_correlation
                ):
                    break

                results.append(
                    {
                        "component": component_id,
                        "n voxel": np.sum(compimg.get_fdata() == component_id),
                        "centroid": [centroid[0:3]],
                        "region": row.Region,
                        "contains": row.Contains,
                        "contained": row.Contained,
                        "correlation": row.Correlation,
                        "max p": row.MaxValue,
                    }
                )
        return pd.DataFrame(results)

    def _plot_input(self, img, niftifile, outdir):
        """plot  image to file"""
        filename = f"{os.path.join(outdir, os.path.basename(niftifile))}.png"
        if not os.path.isfile(filename) or self.overwrite:
            plt.ion()
            fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=self.dpi)
            plotting.plot_glass_brain(img, axes=ax, alpha=0.3)
            fig.tight_layout(pad=0.0)
            plt.ioff()
            fig.savefig(filename, dpi=self.dpi)
        return filename

    def _plot_component(self, arr, component, affine, niftifile, outdir):
        """Plot component to file"""
        filename = (
            f"{os.path.join(outdir, os.path.basename(niftifile))}.{str(component)}.png"
        )
        if not os.path.isfile(filename) or self.overwrite:
            mask = nib.Nifti1Image((arr == component).astype("uint8"), affine)
            fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=self.dpi)
            plt.ion()
            plotting.plot_glass_brain(mask, axes=ax, colorbar=False, alpha=0.3)
            # fig.tight_layout(pad=0.0)
            plt.ioff()
            fig.savefig(filename, dpi=self.dpi)
        return filename

    def _plot_pmap(self, regionname, outdir):
        with QUIET:
            region = self.pmaps.decode_region(regionname)
            pindices = self.pmaps.get_index(regionname)
        assert len(pindices) == 1
        filename = f"{os.path.join(outdir, region.key)}_pmap.png"
        if not os.path.isfile(filename) or self.overwrite:
            pmap = self.pmaps.fetch(pindices[0].map)
            fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=self.dpi)
            plt.ion()
            plotting.plot_glass_brain(
                pmap, axes=ax, colorbar=False, alpha=0.3, cmap="magma"
            )
            # fig.tight_layout(pad=0.0)
            plt.ioff()
            fig.savefig(filename, dpi=self.dpi)
        return filename

    def _plot_profile(self, conn, regionname, outdir):
        with QUIET:
            region = self.pmaps.decode_region(regionname)
        if region not in conn.matrix:
            return None
        filename = f"{os.path.join(outdir, region.key)}_profile.png"
        if not os.path.isfile(filename) or self.overwrite:
            fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=self.dpi)
            plt.ion()
            profile = conn.matrix[region].sort_values(ascending=False)[: self.max_conn]
            profile.plot.bar(grid=True, ax=ax).set_xticklabels(
                [str(_)[:40] for _ in profile.index]
            )
            plt.xticks(rotation=45, fontsize=5, ha="right")
            ax.set_title(f"Streamline counts (top {self.max_conn})", fontsize=10)
            fig.tight_layout(pad=0.2)
            plt.ioff()
            fig.savefig(filename, dpi=self.dpi)
        return filename

    def _build_pdf(
        self,
        assignments,
        niftifile,
        input_plot,
        components_plot,
        component_plots,
        pmap_plots,
        profile_plots,
        outfile,
    ):

        pdf = FPDF()
        plot_height = 40
        text_height = 4
        cell_height = plot_height + text_height

        # title page
        pdf.add_page()
        left = pdf.get_x()
        top = pdf.get_y()

        pdf.set_font("Arial", "BU", 20)
        pdf.set_xy(left, top)
        pdf.cell(40, 10, "Anatomical Assignment")

        pdf.set_font("Arial", "", 10)
        pdf.set_xy(left, top + 14)
        pdf.multi_cell(
            0,
            text_height,
            "\n".join(
                [
                    f"Parcellation: {self.pmaps.parcellation.name}",
                    f"Input file: {niftifile}",
                    f"Found {len(assignments.component.unique())} components",
                    " ",
                    f"For each component, regions with a correlation >{self.min_correlation} are assigned, but at least {self.min_entries}.",
                    " ",
                    f"siibra version {__version__}",
                    f'Computed on {datetime.now().strftime("%c")}',
                ]
            ),
        )

        pdf.set_xy(left, top + 60)
        pdf.image(input_plot, w=180)

        pdf.set_xy(left, top + 60 + 75)
        pdf.image(components_plot, w=180)

        # one page per analyzed component
        components = assignments.component.unique()
        logger.info(f"Building pdf report {outfile} for {len(components)} components.")
        for component in components:

            pdf.add_page()
            pdf.set_font("Arial", "BU", 12)
            pdf.cell(40, text_height, f"Assignments for component #{component}")

            pdf.set_xy(left, 14)
            pdf.image(component_plots[component], h=plot_height)

            selection = assignments[lambda d: d.component == component]

            for i, (index, row) in tqdm(
                enumerate(selection.iterrows()),
                total=len(selection),
                desc=f"- Page #{component}",
                unit="assignments",
            ):

                pdf.set_xy(left, 14 + text_height + (i + 1) * cell_height)
                pdf.image(pmap_plots[row.region], h=plot_height)

                if profile_plots[row.region] is not None:
                    pdf.set_xy(100, 14 + (i + 1) * cell_height)
                    pdf.image(profile_plots[row.region], h=cell_height)

                pdf.set_font("Arial", "B", 10)
                pdf.set_xy(left, 14 + (i + 1) * cell_height)
                pdf.cell(40, text_height, f"{i+1}. {row.region}")

                pdf.set_font("Arial", "", 10)
                pdf.set_xy(left, text_height + 14 + (i + 1) * cell_height)
                txt = "\n".join(
                    [
                        f"Correlation: {row.correlation:.2f}",
                        f"Max. probability: {row['max p']:.2f}",
                    ]
                )
                pdf.multi_cell(0, text_height, txt)

        logger.info(f"Report written to {outfile}")
        pdf.output(outfile, "F")
