import pytest
from unittest.mock import patch, MagicMock
from siibra.features.feature import Feature, ParseLiveQueryIdException, EncodeLiveQueryIdException, space, parcellation
from siibra.core import space, parcellation, region
from siibra.commons import Species

class FooFeatureBase:
    def __init__(self, id) -> None:
        self.id = id
        self.other = "bla"

class FooFeature(FooFeatureBase):
    _live_queries = []

space_inst = space.Space("space_id", "name", Species.UNSPECIFIED_SPECIES)
reg_inst = region.Region("reg_name")
parc_inst = parcellation.Parcellation("parc_id", "name", Species.UNSPECIFIED_SPECIES, regions=(reg_inst,))
feature_inst = FooFeature("feature_id")


list_of_queries = [
    (FooFeatureBase("feature_id"), space_inst, EncodeLiveQueryIdException, None),
    (feature_inst, None, EncodeLiveQueryIdException, None),
    (feature_inst, space_inst, None, f"lq0::FooFeature::s:{space_inst.id}::{feature_inst.id}"),
    (feature_inst, parc_inst, None, f"lq0::FooFeature::p:{parc_inst.id}::{feature_inst.id}"),
    (feature_inst, reg_inst, None, f"lq0::FooFeature::p:{parc_inst.id}::r:{reg_inst.name}::{feature_inst.id}"),
]

@pytest.mark.parametrize("feature,concept,ExCls,expected_id", list_of_queries)
def test_serialize_query_context(feature,concept,ExCls,expected_id):
    if ExCls:
        with pytest.raises(ExCls):
            Feature.serialize_query_context(feature, concept)
        return
    actual_id = Feature.serialize_query_context(feature, concept)
    assert actual_id == expected_id

lq_prefix="lq0"
feature_name="FooFeatures"

@pytest.fixture
def mock_parse_featuretype():
    with patch.object(Feature, "parse_featuretype") as mock:
        mock.return_value = [FooFeature, FooFeatureBase]
        yield mock

@pytest.fixture
def mock_space_registry():
    with patch.object(space.Space, "registry") as mock:
        mock.return_value = MagicMock()
        mock.return_value.__getitem__.return_value = space_inst
        yield mock.return_value
    
@pytest.fixture
def mock_parcellation():
    with patch.object(parcellation.Parcellation, "registry") as mock:
        mock.return_value = MagicMock()
        mock.return_value.__getitem__.return_value = parc_inst
        yield mock.return_value

@pytest.fixture
def mock_region():
    with patch.object(parc_inst, "get_region") as mock:
        mock.return_value = reg_inst
        mock.assert_not_called
        yield mock

@pytest.fixture
def mock_all(mock_parse_featuretype,mock_space_registry,mock_parcellation,mock_region):
    return (mock_parse_featuretype,mock_space_registry,mock_parcellation,mock_region)

def test_mock_featuretype(mock_parse_featuretype):
    feature_types = Feature.parse_featuretype()
    mock_parse_featuretype.assert_called_once()
    assert feature_types == [FooFeature, FooFeatureBase]

    mock_parse_featuretype.return_value = 'foo'
    feature_types = Feature.parse_featuretype()
    assert feature_types == 'foo'
    

def test_mock_registry(mock_space_registry):
    v = space.Space.registry()['foo']
    mock_space_registry.__getitem__.assert_called_with('foo')
    assert v is space_inst

    mock_space_registry.__getitem__.return_value = "bar2"
    v = space.Space.registry()['foo']
    assert v == "bar2"

space_spec = f"s:{space_inst.id}"
parc_spec = f"p:{parc_inst.id}"
reg_spec = f"r:{reg_inst.name}"

list_of_fids = [
    (f"bla::{feature_name}::{space_spec}::{feature_inst.id}",ParseLiveQueryIdException,None,None,None,None),
    (f"{lq_prefix}::{feature_name}::foo:bar::{feature_inst.id}",ParseLiveQueryIdException,None,None,None,None),
    (f"{lq_prefix}::{feature_name}::{reg_spec}::{feature_inst.id}",ParseLiveQueryIdException,None,None,None,None),
    (f"{lq_prefix}::{feature_name}::{space_spec}::{parc_spec}::{feature_inst.id}",ParseLiveQueryIdException,None,None,None,None),
    (f"{lq_prefix}::{feature_name}::{parc_spec}::{space_spec}::{feature_inst.id}",ParseLiveQueryIdException,None,None,None,None),
    (f"{lq_prefix}::{feature_name}::{reg_spec}::{parc_spec}::{feature_inst.id}",ParseLiveQueryIdException,None,None,None,None),
    (f"{lq_prefix}::{feature_name}::{parc_spec}::{feature_inst.id}",None,['parc'],[parc_inst.id],parc_inst,feature_inst.id),
    (f"{lq_prefix}::{feature_name}::{space_spec}::{feature_inst.id}",None,['space'],[space_inst.id],space_inst,feature_inst.id),
    (f"{lq_prefix}::{feature_name}::{parc_spec}::{reg_spec}::{feature_inst.id}",None,['parc', 'region'],[parc_inst.id, reg_inst.name],reg_inst,feature_inst.id),
]

@pytest.mark.parametrize("fid,ExCls,mocks_called,args_used,return_concept,decoded_id", list_of_fids)
def test_deserialize_query_context(fid,ExCls,mocks_called,args_used,return_concept,decoded_id, mock_all):
    mock_parse_featuretype, mock_space_registry, mock_parcellation, mock_region = mock_all
    if ExCls:
        with pytest.raises(ExCls):
            Feature.deserialize_query_context(fid)
        return
    
    F, concept, fid = Feature.deserialize_query_context(fid)

    mock_parse_featuretype.assert_called_once()

    for (key, mock) in [("space", mock_space_registry.__getitem__), ("parc", mock_parcellation.__getitem__), ("region", mock_region)]:
        if key in mocks_called:
            for idx, v in enumerate(mocks_called):
                if v == key:
                    mock.assert_called_once_with(args_used[idx])
                    break
            else:
                assert False, f"not called"
        else:
            mock.assert_not_called()
    
    assert concept is return_concept
    assert fid == decoded_id
    assert F is FooFeature


def test_wrap_feature():
    new_feat = Feature.wrap_livequery_feature(feature_inst, "helloworld")
    assert new_feat.other == feature_inst.other
    assert new_feat.id != feature_inst.id
    assert new_feat.id == "helloworld"
    assert new_feat.__class__ is FooFeature

