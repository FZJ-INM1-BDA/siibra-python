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

from .feature import RegionalFeature,GlobalFeature
from .extractor import FeatureExtractor
import requests
import json
import os
import re
from .. import retrieval, ebrains

kg_feature_query_kwargs={
    'params': {
        'vocab': 'https://schema.hbp.eu/myQuery/'
    }
}

kg_feature_summary_kwargs={
    'org': 'minds',
    'domain': 'core',
    'schema': 'parcellationregion',
    'version': 'v1.0.0'
}

kg_feature_full_kwargs={
    'org': 'minds',
    'domain': 'core',
    'schema': 'dataset',
    'version': 'v1.0.0'
}

kg_feature_summary_spec={
    "@context": {
        "@vocab": "https://schema.hbp.eu/graphQuery/",
        "query": "https://schema.hbp.eu/myQuery/",
        "fieldname": {
            "@id": "fieldname",
            "@type": "@id"
        },
        "merge": {
            "@type": "@id",
            "@id": "merge"
        },
        "relative_path": {
            "@id": "relative_path",
            "@type": "@id"
        }
    },
    "fields": [
        {
            "fieldname": "query:@id",
            "relative_path": "@id"
        },
        {
            "fieldname": "query:identifier",
            "relative_path": "http://schema.org/identifier"
        },
        {
            "fieldname": "query:name",
            "relative_path": "http://schema.org/name"
        },
        {
            "fieldname": "query:datasets",
            "relative_path": {
                "@id":"https://schema.hbp.eu/minds/parcellationRegion",
                "reverse": True
            },
            "fields": [
                {
                  "fieldname": "query:@id",
                  "relative_path": "@id"
                },
                {
                  "fieldname": "query:identifier",
                  "relative_path": "http://schema.org/identifier"
                },
                {
                  "fieldname": "query:name",
                  "relative_path": "http://schema.org/name"
                },
                {
                  "fieldname": "query:embargo_status",
                  "relative_path": "https://schema.hbp.eu/minds/embargo_status",
                  "fields": [
                    {
                      "fieldname": "query:@id",
                      "relative_path": "@id"
                    },
                    {
                      "fieldname": "query:identifier",
                      "relative_path": "http://schema.org/identifier"
                    },
                    {
                      "fieldname": "query:name",
                      "relative_path": "http://schema.org/name"
                    }
                  ]
                }
            ]
        }
    ]
}

kg_feature_full_spec={
  "@context": {
    "fieldname": {
      "@type": "@id",
      "@id": "fieldname"
    },
    "@vocab": "https://schema.hbp.eu/graphQuery/",
    "merge": {
      "@type": "@id",
      "@id": "merge"
    },
    "query": "https://schema.hbp.eu/myQuery/",
    "relative_path": {
      "@type": "@id",
      "@id": "relative_path"
    }
  },
  "fields": [
    {
      "fieldname": "query:activity",
      "relative_path": "https://schema.hbp.eu/minds/activity",
      "fields": [
        {
          "fieldname": "query:preparation",
          "relative_path": [
            "https://schema.hbp.eu/minds/preparation",
            "http://schema.org/name"
          ]
        },
        {
          "fieldname": "query:protocols",
          "relative_path": [
            "https://schema.hbp.eu/minds/protocols",
            "http://schema.org/name"
          ]
        }
      ]
    },
    {
      "fieldname": "query:kgReference",
      "relative_path": [
        {
          "@id": "https://schema.hbp.eu/minds/doireference",
          "reverse": True
        },
        "https://schema.hbp.eu/minds/doi"
      ]
    },
    {
      "fieldname": "query:name",
      "relative_path": "http://schema.org/name"
    },
    {
      "fieldname": "query:description",
      "relative_path": "http://schema.org/description"
    },
    {
      "fieldname": "query:contributors",
      "relative_path": "https://schema.hbp.eu/minds/contributors",
      "fields": [
        {
          "fieldname": "query:name",
          "relative_path": "http://schema.org/name"
        },
        {
          "fieldname": "query:@id",
          "relative_path": "@id"
        },
        {
          "fieldname": "query:schema.org/shortName",
          "relative_path": "http://schema.org/shortName"
        },
        {
          "fieldname": "query:shortName",
          "relative_path": "https://schema.hbp.eu/minds/shortName"
        },
        {
          "fieldname": "query:identifier",
          "relative_path": "http://schema.org/identifier"
        }
      ]
    },
    {
      "fieldname": "query:formats",
      "relative_path": [
        "https://schema.hbp.eu/minds/formats",
        "http://schema.org/name"
      ]
    },
    {
      "fieldname": "query:referenceSpaces",
      "relative_path": "https://schema.hbp.eu/minds/reference_space",
      "fields": [
        {
          "fieldname": "query:name",
          "relative_path": "http://schema.org/name"
        },
        {
          "fieldname": "query:fullId",
          "relative_path": "@id"
        }
      ]
    },
    {
      "fieldname": "query:license",
      "relative_path": "https://schema.hbp.eu/minds/license",
      "fields": [
        {
          "fieldname": "query:name",
          "relative_path": "http://schema.org/name"
        },
        {
          "fieldname": "query:relativeUrl",
          "relative_path": "https://schema.hbp.eu/relativeUrl"
        }
      ]
    },
    {
      "fieldname": "query:licenseInfo",
      "relative_path": "https://schema.hbp.eu/minds/license_info",
      "fields": [
        {
          "fieldname": "query:name",
          "relative_path": "http://schema.org/name"
        },
        {
          "fieldname": "query:url",
          "relative_path": "http://schema.org/url"
        }
      ]
    },
    {
      "fieldname": "query:publications",
      "relative_path": "https://schema.hbp.eu/minds/publications",
      "fields": [
        {
          "fieldname": "query:doi",
          "relative_path": "https://schema.hbp.eu/minds/doi"
        },
        {
          "fieldname": "query:cite",
          "relative_path": "https://schema.hbp.eu/minds/cite"
        },
        {
          "fieldname": "query:name",
          "relative_path": "http://schema.org/name"
        }
      ]
    },
    {
      "fieldname": "query:parcellationRegion",
      "relative_path": "https://schema.hbp.eu/minds/parcellationRegion",
      "fields": [
        {
          "fieldname": "query:fullId",
          "relative_path": "@id"
        },
        {
          "fieldname": "query:name",
          "relative_path": "http://schema.org/name"
        },
        {
          "fieldname": "query:species",
          "relative_path": "https://schema.hbp.eu/minds/species",
          "fields": [
            {
              "fieldname": "query:name",
              "relative_path": "http://schema.org/name"
            },
            {
              "fieldname": "query:@id",
              "relative_path": "@id"
            },
            {
              "fieldname": "query:identifier",
              "relative_path": "http://schema.org/identifier"
            }
          ]
        },
        {
          "fieldname": "query:alias",
          "relative_path": "https://schema.hbp.eu/minds/alias"
        }
      ]
    },
    {
      "fieldname": "query:custodians",
      "relative_path": "https://schema.hbp.eu/minds/owners",
      "fields": [
        {
          "fieldname": "query:name",
          "relative_path": "http://schema.org/name"
        },
        {
          "fieldname": "query:@id",
          "relative_path": "@id"
        },
        {
          "fieldname": "query:schema.org/shortName",
          "relative_path": "http://schema.org/shortName"
        },
        {
          "fieldname": "query:shortName",
          "relative_path": "https://schema.hbp.eu/minds/shortName"
        },
        {
          "fieldname": "query:identifier",
          "relative_path": "http://schema.org/identifier"
        }
      ]
    },
    {
      "fieldname": "query:embargoStatus",
      "relative_path": "https://schema.hbp.eu/minds/embargo_status",
      "fields": [
        {
          "fieldname": "query:name",
          "relative_path": "http://schema.org/name"
        },
        {
          "fieldname": "query:@id",
          "relative_path": "@id"
        },
        {
          "fieldname": "query:identifier",
          "relative_path": "http://schema.org/identifier"
        }
      ]
    },
    {
      "fieldname": "query:species",
      "relative_path": [
        "https://schema.hbp.eu/minds/specimen_group",
        "https://schema.hbp.eu/minds/subjects",
        "https://schema.hbp.eu/minds/species",
        "http://schema.org/name"
      ]
    },
    {
      "fieldname": "query:methods",
      "merge": [
        {
          "relative_path": [
            "https://schema.hbp.eu/minds/activity",
            "https://schema.hbp.eu/minds/methods",
            "http://schema.org/name"
          ]
        },
        {
          "relative_path": [
            "https://schema.hbp.eu/minds/specimen_group",
            "https://schema.hbp.eu/minds/subjects",
            "https://schema.hbp.eu/minds/samples",
            "https://schema.hbp.eu/minds/methods",
            "http://schema.org/name"
          ]
        }
      ]
    },
    {
      "fieldname": "query:project",
      "relative_path": [
        "https://schema.hbp.eu/minds/component",
        "http://schema.org/name"
      ]
    },
    {
      "fieldname": "query:datasetDOI",
      "relative_path": "https://schema.hbp.eu/minds/publications",
      "fields": [
        {
          "fieldname": "query:doi",
          "relative_path": "https://schema.hbp.eu/minds/doi"
        },
        {
          "fieldname": "query:cite",
          "relative_path": "https://schema.hbp.eu/minds/cite"
        }
      ]
    },
    {
      "fieldname": "query:files",
      "merge": [
        {
          "relative_path": {
            "@id": "minds/core/fileassociation/v1.0.0",
            "reverse": True
          }
        },
        {
          "relative_path": [
            "https://schema.hbp.eu/minds/specimen_group",
            "https://schema.hbp.eu/minds/subjects",
            "https://schema.hbp.eu/minds/samples",
            {
              "@id": "minds/core/fileassociation/v1.0.0",
              "reverse": True
            }
          ]
        }
      ],
      "fields": [
        {
          "fieldname": "query:name",
          "relative_path": "http://schema.org/name"
        },
        {
          "fieldname": "query:absolutePath",
          "relative_path": "https://schema.hbp.eu/cscs/absolute_path"
        },
        {
          "fieldname": "query:byteSize",
          "relative_path": "https://schema.hbp.eu/cscs/byte_size"
        },
        {
          "fieldname": "query:contentType",
          "relative_path": "https://schema.hbp.eu/cscs/content_type"
        }
      ]
    },
    {
      "fieldname": "query:fullId",
      "relative_path": "@id"
    },
    {
      "fieldname": "query:id",
      "relative_path": "http://schema.org/identifier"
    },
    {
      "fieldname": "query:parcellationAtlas",
      "relative_path": "https://schema.hbp.eu/minds/parcellationAtlas",
      "fields": [
        {
          "fieldname": "query:name",
          "relative_path": "http://schema.org/name"
        },
        {
          "fieldname": "query:fullId",
          "relative_path": "@id"
        },
        {
          "fieldname": "query:id",
          "relative_path": "http://schema.org/identifier"
        }
      ]
    }
  ]
}

KG_REGIONAL_FEATURE_SUMMARY_QUERY_NAME = 'siibra-kg-feature-summary-0.0.1'
KG_REGIONAL_FEATURE_FULL_QUERY_NAME='interactiveViewerKgQuery-v1_0'
class KgRegionalFeature(RegionalFeature):
    def __init__(self, region, id, name, embargo_status):
        self.region = region
        self.id = id
        self.name = name
        self.embargo_status = embargo_status
        self._detail = None

    @property
    def detail(self):
        if not self._detail:
            self._load()
        return self._detail

    def _load(self):
        if self.id is None:
            raise Exception('id is required')
        match=re.search(r"\/([a-f0-9-]+)$", self.id)
        if not match:
            raise Exception('id cannot be parsed properly')
        instance_id=match.group(1)
        result=ebrains.execute_query_by_id(query_id=KG_REGIONAL_FEATURE_FULL_QUERY_NAME, instance_id=instance_id,
            **kg_feature_query_kwargs,**kg_feature_full_kwargs)
        self._detail = result

    def __str__(self):
        return self.name


class KgRegionalFeatureExtractor(FeatureExtractor):
    _FEATURETYPE=KgRegionalFeature
    def __init__(self, atlas):
        FeatureExtractor.__init__(self,atlas)
        
        # potentially, using kg_id is a lot quicker
        # but if user selects region with higher level of hierarchy, this benefit may be offset by numerous http calls
        # even if they were cached...
        # kg_id=atlas.selected_region.attrs.get('fullId', {}).get('kg', {}).get('kgId', None)

        result=ebrains.execute_query_by_id(query_id=KG_REGIONAL_FEATURE_SUMMARY_QUERY_NAME,
            **kg_feature_query_kwargs,**kg_feature_summary_kwargs)

        kg_regional_features=[KgRegionalFeature(
            region=atlas.find_regions(r.get('name', None)),
            id=dataset.get('@id'),
            name=dataset.get('name'),
            embargo_status=dataset.get('embargo_status')
        ) for r in result.get('results', []) for dataset in r.get('datasets', []) ]

        for f in kg_regional_features:
            self.register(f)


def set_specs():
    req = ebrains.upload_schema(query_id=KG_REGIONAL_FEATURE_SUMMARY_QUERY_NAME, spec=kg_feature_summary_spec,
        **kg_feature_summary_kwargs)
    assert req.status_code < 400

if __name__ == '__main__':
    pass
