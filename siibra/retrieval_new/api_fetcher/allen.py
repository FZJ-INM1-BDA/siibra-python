from ...assignment.assignment import register_collection_generator
from ...concepts.feature import Feature
from ...concepts.attribute_collection import AttributeCollection
from ...descriptions import Gene, RegionSpec, Modality
from ...descriptions.modality import register_modalities

modality_of_interest = Modality(value="Gene Expressions")

@register_modalities()
def add_allen_modality():
    yield modality_of_interest

@register_collection_generator(Feature)
def query_allen_gene_api(input: AttributeCollection):
    if not modality_of_interest in input.get(Modality):
        return []
    genes = input.get(Gene)
    if len(genes) == 0:
        return []
    region_specs = input.get(RegionSpec)
    return []
