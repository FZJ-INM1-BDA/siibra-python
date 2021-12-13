from siibra.core.datasets import EbrainsDataset
from siibra.retrieval import EbrainsRequest
from unittest.mock import patch

legacy_dict={
  "formats": [
    "Tagged Image File Format (TIFF, TIF)",
    "Tab-Separated Values (TSV)"
  ],
  "datasetDOI": [
    {
      "cite": "Eickhoff, S. B., Schleicher, A., Scheperjans, F., Palomero-Gallagher, N., & Zilles, K. (2007). Analysis of neurotransmitter receptor distribution patterns in the cerebral cortex. NeuroImage, 34(4), 1317–1330. ",
      "doi": "10.1016/j.neuroimage.2006.11.016"
    },
    {
      "cite": "Zilles, K., Bacha-Trams, M., Palomero-Gallagher, N., Amunts, K., & Friederici, A. D. (2015). Common molecular basis of the sentence comprehension network revealed by neurotransmitter receptor fingerprints. Cortex, 63, 79–89. ",
      "doi": "10.1016/j.cortex.2014.07.007"
    }
  ],
  "activity": [
    {
      "protocols": [
        "histology",
        "brain mapping"
      ],
      "preparation": [
        "Ex vivo"
      ]
    }
  ],
  "referenceSpaces": [],
  "methods": [
    "Semantic atlas registration",
    "autoradiography - quantitative analysis",
    "autoradiography - imaging"
  ],
  "custodians": [
    {
      "schema.org/shortName": "Palomero-Gallagher, N.",
      "identifier": "3ac753a73ac8cfd3151639808ae45913",
      "name": "Palomero-Gallagher, Nicola",
      "@id": "https://nexus.humanbrainproject.org/v0/data/minds/core/person/v1.0.0/d77176e3-faa9-4c93-8dc0-7f3663c6d2ee",
      "shortName": "Palomero-Gallagher, N."
    },
    {
      "schema.org/shortName": "Zilles, K.",
      "identifier": "2457b4a7cf0b3fa199dcc3c88180dc9f",
      "name": "Zilles, Karl",
      "@id": "https://nexus.humanbrainproject.org/v0/data/minds/core/person/v1.0.0/bacae1fc-d8ff-4050-8388-7d826592c62c",
      "shortName": "Zilles, K."
    }
  ],
  "project": [
    "Quantitative receptor data"
  ],
  "description": "This dataset contains the densities (in fmol/mg protein) of receptors for classical neurotransmitters in Area hOc1 (V1, 17, CalcS) obtained by means of quantitative _in vitro_ autoradiography. The receptor densities are visualized as _fingerprints_ (**fp**), which provide the mean density and standard deviation for each of the analyzed receptor types, averaged across samples. \n\nOverview of available measurements [ **receptor** \\| **_neurotransmitter_** \\| _labeling agent_ ]:\n**AMPA** \\| **_glutamate_** \\| _[<sup>3</sup>H]AMPA_\n**kainate** \\| **_glutamate_** \\| _[<sup>3</sup>H]kainate_\n**NMDA** \\| **_glutamate_** \\| _[<sup>3</sup>H]MK-801_\n**GABA<sub>A</sub>** \\| **_GABA_** \\| _[<sup>3</sup>H]muscimol_\n**GABA<sub>B</sub>** \\| **_GABA_** \\| _[<sup>3</sup>H]CGP54626_\n**GABA<sub>A</sub>/BZ** \\| **_GABA_** \\| _[<sup>3</sup>H]flumazenil_\n**muscarinic M<sub>1</sub>** \\| **_acetylcholine_** \\| _[<sup>3</sup>H]pirenzepine_\n**muscarinic M<sub>2</sub>** \\| **_acetylcholine_** \\| _[<sup>3</sup>H]oxotremorine-M_\n**muscarinic M<sub>3</sub>** \\| **_acetylcholine_** \\| _[<sup>3</sup>H]4-DAMP_\n**nicotinic &#945<sub>4</sub>&#946<sub>2</sub>** \\| **_acetylcholine_** \\| _[<sup>3</sup>H]epibatidine_\n**&#945<sub>1</sub>** \\| **_noradrenalin/norepinephrine_** \\| _[<sup>3</sup>H]prazosin_\n**&#945<sub>2</sub>** \\| **_noradrenalin/norepinephrine_** \\| _[<sup>3</sup>H]UK-14,304_\n**5-HT<sub>1A</sub>** \\| **_serotonin_** \\| _[<sup>3</sup>H]8-OH-DPAT_\n**5-HT<sub>2</sub>** \\| **_serotonin_** \\| _[<sup>3</sup>H]ketanserin_\n**D<sub>1</sub>** \\| **_dopamine_** \\| _[<sup>3</sup>H]SCH23390_\n\nFor exemplary samples, we also provide laminar _profiles_ (**pr**)  and/or color-coded laminar _autoradiography_ images (**ar**). The density profiles provide, for a single tissue sample, an exemplary density distribution for a single receptor from the pial surface to the border between layer VI and the white matter. The autoradiography images show an exemplary density distribution of a single receptor for one laminar cross-section in a single tissue sample. \n\nInformation on the used tissue samples and corresponding subjects for the receptor fingerprints, profiles and autoradiographs as well as a list of analyzed receptors accompanies the provided dataset.\n\n**For methodological details, see:**\nZilles, K. et al. (2002). Quantitative analysis of cyto- and receptorarchitecture of the human brain, pp. 573-602. In: Brain Mapping: The Methods, 2nd edition (A.W. Toga and J.C. Mazziotta, eds.). San Diego, Academic Press.\n\nPalomero-Gallagher N, Zilles K. (2018) Cyto- and receptorarchitectonic mapping of the human brain. In: Handbook of Clinical Neurology 150: 355-387",
  "parcellationAtlas": [
    {
      "name": "Julich-Brain Atlas",
      "fullId": "https://nexus.humanbrainproject.org/v0/data/minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579",
      "id": [
        "deec923ec31a82f89a9c7c76a6fefd6b",
        "e2d45e028b6da0f6d9fdb9491a4de80a"
      ]
    }
  ],
  "licenseInfo": [
    {
      "name": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International",
      "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/"
    }
  ],
  "embargoStatus": [
    {
      "identifier": [
        "b24ce0cd392a5b0b8dedc66c25213594",
        "b24ce0cd392a5b0b8dedc66c25213594"
      ],
      "name": "Free",
      "@id": "https://nexus.humanbrainproject.org/v0/data/minds/core/embargostatus/v1.0.0/222b535c-2e8f-4892-acf4-39006c5219b9"
    }
  ],
  "license": [],
  "parcellationRegion": [
    {
      "species": [],
      "name": "Area hOc1 (V1, 17, CalcS)",
      "alias": "Area hOc1 (V1, 17, CalcS)",
      "fullId": "https://nexus.humanbrainproject.org/v0/data/minds/core/parcellationregion/v1.0.0/5151ab8f-d8cb-4e67-a449-afe2a41fb007"
    }
  ],
  "species": [
    "Homo sapiens"
  ],
  "name": "Density measurements of different receptors for Area hOc1 (V1, 17, CalcS) [human, v1.0]",
  "files": [
    {
      "byteSize": 1802,
      "name": "hOc1_pr_D1.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/D1/hOc1_pr_D1.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 59,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/alpha4beta2/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 1682,
      "name": "receptors.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/receptors.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/M3/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/5-HT1A/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/M2/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/D1/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 1802,
      "name": "hOc1_pr_M1.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/M1/hOc1_pr_M1.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/GABAA/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 3591641,
      "name": "hOc1_ar_NMDA.tif",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/NMDA/hOc1_ar_NMDA.tif",
      "contentType": "image/tiff"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/5-HT1A/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 3633649,
      "name": "hOc1_ar_5-HT1A.tif",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/5-HT1A/hOc1_ar_5-HT1A.tif",
      "contentType": "image/tiff"
    },
    {
      "byteSize": 1851,
      "name": "hOc1_pr_alpha2.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/alpha2/hOc1_pr_alpha2.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 999,
      "name": "labeling-agents.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/labeling-agents.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/kainate/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 1701,
      "name": "hOc1_pr_alpha4beta2.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/alpha4beta2/hOc1_pr_alpha4beta2.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 3618025,
      "name": "hOc1_ar_M2.tif",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/M2/hOc1_ar_M2.tif",
      "contentType": "image/tiff"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/kainate/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 84,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/NMDA/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 1802,
      "name": "hOc1_pr_M2.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/M2/hOc1_pr_M2.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/GABAB/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/alpha1/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 1748,
      "name": "hOc1_pr_alpha1.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/alpha1/hOc1_pr_alpha1.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 84,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/5-HT1A/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/M1/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/AMPA/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/M2/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 1903,
      "name": "hOc1_pr_BZ.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/BZ/hOc1_pr_BZ.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/M3/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 1833,
      "name": "hOc1_pr_M3.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/M3/hOc1_pr_M3.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 1835,
      "name": "hOc1_pr_kainate.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/kainate/hOc1_pr_kainate.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/alpha4beta2/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 3627611,
      "name": "hOc1_ar_BZ.tif",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/BZ/hOc1_ar_BZ.tif",
      "contentType": "image/tiff"
    },
    {
      "byteSize": 1802,
      "name": "hOc1_pr_5-HT2.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/5-HT2/hOc1_pr_5-HT2.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/D1/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/alpha2/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 99,
      "name": "subjects_all.csv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/subjects_all.csv",
      "contentType": "text/csv"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/M3/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 1761,
      "name": "hOc1_pr_5-HT1A.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/5-HT1A/hOc1_pr_5-HT1A.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/NMDA/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/GABAA/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/alpha1/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/5-HT2/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/BZ/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 161,
      "name": "tissue-samples_all.csv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/tissue-samples_all.csv",
      "contentType": "text/csv"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/GABAB/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/mGluR2_3/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 3609567,
      "name": "hOc1_ar_M1.tif",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/M1/hOc1_ar_M1.tif",
      "contentType": "image/tiff"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/mGluR2_3/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/5-HT2/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 59,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/NMDA/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/5-HT2/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 3584447,
      "name": "hOc1_ar_5-HT2.tif",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/5-HT2/hOc1_ar_5-HT2.tif",
      "contentType": "image/tiff"
    },
    {
      "byteSize": 999,
      "name": "labeling-agents.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/labeling-agents.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/BZ/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/M1/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 59,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/D1/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 1682,
      "name": "receptors.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/receptors.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 3613223,
      "name": "hOc1_ar_GABAB.tif",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/GABAB/hOc1_ar_GABAB.tif",
      "contentType": "image/tiff"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/alpha1/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 84,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/AMPA/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 3619237,
      "name": "hOc1_ar_AMPA.tif",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/AMPA/hOc1_ar_AMPA.tif",
      "contentType": "image/tiff"
    },
    {
      "byteSize": 3614399,
      "name": "hOc1_ar_mGluR2_3.tif",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/mGluR2_3/hOc1_ar_mGluR2_3.tif",
      "contentType": "image/tiff"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/alpha1/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 84,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/M1/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 1680,
      "name": "receptors.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/receptors.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/alpha4beta2/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 84,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/alpha2/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/NMDA/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 3632461,
      "name": "hOc1_ar_alpha4beta2.tif",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/alpha4beta2/hOc1_ar_alpha4beta2.tif",
      "contentType": "image/tiff"
    },
    {
      "byteSize": 3652947,
      "name": "hOc1_ar_alpha1.tif",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/alpha1/hOc1_ar_alpha1.tif",
      "contentType": "image/tiff"
    },
    {
      "byteSize": 3640855,
      "name": "hOc1_ar_M3.tif",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/M3/hOc1_ar_M3.tif",
      "contentType": "image/tiff"
    },
    {
      "byteSize": 59,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/AMPA/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 1879,
      "name": "hOc1_pr_GABAA.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/GABAA/hOc1_pr_GABAA.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 3590413,
      "name": "hOc1_ar_GABAA.tif",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/GABAA/hOc1_ar_GABAA.tif",
      "contentType": "image/tiff"
    },
    {
      "byteSize": 84,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/BZ/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/M3/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/kainate/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 1802,
      "name": "hOc1_pr_NMDA.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/NMDA/hOc1_pr_NMDA.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 59,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/kainate/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/GABAA/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 1004,
      "name": "labeling-agents.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/labeling-agents.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/M1/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 84,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/alpha4beta2/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/BZ/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 3645717,
      "name": "hOc1_ar_kainate.tif",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/kainate/hOc1_ar_kainate.tif",
      "contentType": "image/tiff"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/D1/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/M2/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 3620417,
      "name": "hOc1_ar_alpha2.tif",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/alpha2/hOc1_ar_alpha2.tif",
      "contentType": "image/tiff"
    },
    {
      "byteSize": 1802,
      "name": "hOc1_pr_AMPA.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/AMPA/hOc1_pr_AMPA.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/M2/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/alpha2/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/GABAB/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 1895,
      "name": "hOc1_pr_GABAB.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/GABAB/hOc1_pr_GABAB.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 59,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/mGluR2_3/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 84,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/mGluR2_3/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/alpha2/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/GABAA/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 389,
      "name": "hOc1_fp_20171202.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_fp_20171202.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 1875,
      "name": "hOc1_pr_mGluR2_3.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/mGluR2_3/hOc1_pr_mGluR2_3.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/5-HT1A/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 61,
      "name": "subjects.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/AMPA/subjects.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 3648061,
      "name": "hOc1_ar_D1.tif",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/D1/hOc1_ar_D1.tif",
      "contentType": "image/tiff"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_pr_examples/5-HT2/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    },
    {
      "byteSize": 83,
      "name": "tissue-samples.tsv",
      "absolutePath": "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000045_receptors-human-hOc1_pub/v1.0/hOc1_ar_examples/GABAB/tissue-samples.tsv",
      "contentType": "text/tab-separated-values"
    }
  ],
  "fullId": "https://nexus.humanbrainproject.org/v0/data/minds/core/dataset/v1.0.0/e715e1f7-2079-45c4-a67f-f76b102acfce",
  "contributors": [
    {
      "schema.org/shortName": "Eickhoff, S. B.",
      "identifier": "f7e179682ccf98378bb566852f442c93",
      "name": "Eickhoff, Simon B.",
      "@id": "https://nexus.humanbrainproject.org/v0/data/minds/core/person/v1.0.0/99a95057-bd4d-43ab-ad70-b7920ceedc48",
      "shortName": "Eickhoff, S. B."
    },
    {
      "schema.org/shortName": "Amunts, K.",
      "identifier": "e86dc72d5594a43f7a1db1e3945db2bf",
      "name": "Amunts, Katrin",
      "@id": "https://nexus.humanbrainproject.org/v0/data/minds/core/person/v1.0.0/01784c79-9a7b-4b47-83b6-0f50c075af81",
      "shortName": "Amunts, K."
    },
    {
      "schema.org/shortName": "Palomero-Gallagher, N.",
      "identifier": "3ac753a73ac8cfd3151639808ae45913",
      "name": "Palomero-Gallagher, Nicola",
      "@id": "https://nexus.humanbrainproject.org/v0/data/minds/core/person/v1.0.0/d77176e3-faa9-4c93-8dc0-7f3663c6d2ee",
      "shortName": "Palomero-Gallagher, N."
    },
    {
      "schema.org/shortName": "Zilles, K.",
      "identifier": "2457b4a7cf0b3fa199dcc3c88180dc9f",
      "name": "Zilles, Karl",
      "@id": "https://nexus.humanbrainproject.org/v0/data/minds/core/person/v1.0.0/bacae1fc-d8ff-4050-8388-7d826592c62c",
      "shortName": "Zilles, K."
    }
  ],
  "id": "0616d1e97b8be75de526bc265d9af540",
  "kgReference": [
    "10.25493/P8SD-JMH"
  ],
  "publications": [
    {
      "name": "Analysis of neurotransmitter receptor distribution patterns in the cerebral cortex",
      "cite": "Eickhoff, S. B., Schleicher, A., Scheperjans, F., Palomero-Gallagher, N., & Zilles, K. (2007). Analysis of neurotransmitter receptor distribution patterns in the cerebral cortex. NeuroImage, 34(4), 1317–1330. ",
      "doi": "10.1016/j.neuroimage.2006.11.016"
    },
    {
      "name": "Common molecular basis of the sentence comprehension network revealed by neurotransmitter receptor fingerprints",
      "cite": "Zilles, K., Bacha-Trams, M., Palomero-Gallagher, N., Amunts, K., & Friederici, A. D. (2015). Common molecular basis of the sentence comprehension network revealed by neurotransmitter receptor fingerprints. Cortex, 63, 79–89. ",
      "doi": "10.1016/j.cortex.2014.07.007"
    }
  ]
}

def test_ebrains_dataset():
    with patch.object(EbrainsRequest, 'get', return_value=legacy_dict) as mocked_get:
        
        ebrains_dataset = EbrainsDataset.parse_legacy({
            "id": "e715e1f7-2079-45c4-a67f-f76b102acfce"
        })
        mocked_get.assert_not_called()
        ebrains_dataset.json()
        ebrains_dataset.json()
        mocked_get.assert_called_once()

