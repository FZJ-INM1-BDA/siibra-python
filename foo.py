from typing import Type
from itertools import product

import siibra
from siibra.concepts.attribute_collection import AttributeCollection
from siibra.concepts.feature import Feature
from siibra.assignment.assignment import get, match
from siibra.descriptions import Modality, RegionSpec, Gene
from siibra.concepts.query_parameter import QueryParam
from siibra.atlases import Region
import siibra.descriptions
from siibra.dataitems import Image
from siibra.exceptions import UnregisteredAttrCompException
from siibra.locations import BBox

region_spec = RegionSpec(value="Area hOc1 (V1, 17, CalcS)")
modality = Modality(value="Neurotransmitter receptor density fingerprint")

# query by regionspec... works?
query1 = QueryParam(attributes=[region_spec])
features1 = list(get(query1, Feature))
print(len(features1))

# query by modality works
query2 = QueryParam(attributes=[modality])
features2 = list(get(query2, Feature))
print(len(features2))

# but combining them does not work
# since the check goes by any check
query3 = QueryParam(attributes=[
    region_spec,
    modality,
])
features3 = list(get(query3, Feature))
print(len(features3))


# I suspect we will have to introduce helper methods like below to achieve `and` check
def feature_get(qp: QueryParam):
    col_type = Feature
    def _filter(_qp: QueryParam, feat: AttributeCollection) -> bool:
        new_qps = list(_qp.split_attrs())
        new_qps = sorted(new_qps,
                         key=lambda b: len([attr for attr in b.attributes if isinstance(attr, Modality)]),
                         reverse=True)
        for new_qp in new_qps:
            try:
                if not match(new_qp, feat):
                    return False
            except UnregisteredAttrCompException:
                continue
        return True
    
    for feat in get(qp, col_type):
        if _filter(qp, feat):
            yield feat

features4 = list(feature_get(query3))
print(len(features4))


# testing modality autocomplete
# works at runtime (not statically)
print(
    "testing modality dir",
    siibra.descriptions.modality.vocab.__dir__()
)

assert siibra.descriptions.modality.vocab.GENE_EXPRESSIONS == siibra.descriptions.modality.Modality(value="Gene Expressions")

mni152_julichbrain_29_hoc1_lh_pmap = Image(format="nii", space_id="minds/core/referencespace/v1.0.0/dafcffc5-4826-4bf1-8ff6-46b8a31ff8e2",
                                    url="https://neuroglancer.humanbrainproject.eu/precomputed/data-repo-ng-bot/20210616-julichbrain-v2.9.0-complete-mpm/PMs/Area-hOc1/4.2/Area-hOc1_l_N10_nlin2ICBM152asym2009c_4.2_publicP_026bcbe494dc4bfe702f2b1cc927a7c1.nii.gz")

region = Region(attributes=[mni152_julichbrain_29_hoc1_lh_pmap])
gene_maoa = Gene(value="MAOA")
gene_tac1 = Gene(value="TAC1")
query = QueryParam(attributes=[mni152_julichbrain_29_hoc1_lh_pmap,
                               gene_maoa,
                               gene_tac1,
                               siibra.descriptions.modality.vocab.GENE_EXPRESSIONS ])

gene_expression = list(feature_get(query))
print(gene_expression)


# # testing querying via bounding box
cell_body_staining = siibra.descriptions.modality.vocab.CELL_BODY_STAINING
bbox = BBox(minpoint=[-11, -11, -11],
            maxpoint=[11, 11, 11],
            space_id="minds/core/referencespace/v1.0.0/a1655b99-82f1-420f-a3c2-fe80fd4c8588")
query = QueryParam(attributes=[cell_body_staining,
                               bbox,])
cell_body_staining_features = list(feature_get(query))
assert len(cell_body_staining_features) > 0
print(f"found {len(cell_body_staining_features)} cell body staining features")


# querying feature that's both semantically anchored
# as well as locationally anchored

modalities = (
    siibra.descriptions.modality.vocab.SEGMENTED_CELL_BODY_DENSITY,
    siibra.descriptions.modality.vocab.CELL_BODY_DENSITY,
)

hoc1_region_spec = RegionSpec(value="Area hOc1 (V1, 17, CalcS)")
bb_hoc1_lh_labelled_map = Image(format="neuroglancer/precomputed", space_id="minds/core/referencespace/v1.0.0/a1655b99-82f1-420f-a3c2-fe80fd4c8588",
                                url="https://neuroglancer.humanbrainproject.eu/precomputed/BigBrainRelease.2015/2022_map_release/hOc1_left")
bbhoc1_bbox = bb_hoc1_lh_labelled_map.boundingbox
spatial_attr = (
    hoc1_region_spec,
    bbhoc1_bbox,
    # TODO use Image directly (need to implement Image.data for neuroglancer)
)

for modality, spec in product(modalities, spatial_attr):
    print("------------")
    print(f"Matching {modality=} and {spec=}")
    query = QueryParam(attributes=[modality, spec])
    features = list(feature_get(query))
    print(f"found {len(features)=}")

mod = siibra.descriptions.modality.vocab.MODIFIED_SILVER_STAINING
bbox = bbhoc1_bbox

features = list(feature_get(QueryParam(attributes=[mod, bbox])))
assert len(features) > 0
for f in features:
    assert len(f.attributes) > 0

