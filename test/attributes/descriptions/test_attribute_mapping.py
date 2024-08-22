# Copyright 2018-2024
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

import pytest

from siibra.attributes.descriptions.attribute_mapping import AttributeMapping


@pytest.fixture
def region_mapping():
    yield AttributeMapping(
        parcellation_id="foo-bar",
        region_mapping={
            "foo-region-0": [{"@type": "volume/ref"}],
            "foo-region-1": [
                {"@type": "volume/ref", "target": "foo-target-1"},
                {"@type": "volume/ref", "target": "foo-target-2"},
            ],
            "foo-region-2": [
                {"@type": "volume/ref", "target": "foo-target-1"},
                {"@type": "volume/ref", "target": "foo-target-2"},
            ],
            "foo-region-4": [],
            "foo-region-5": [
                {"@type": "volume/ref", "target": "foo-target-3"},
                {"@type": "volume/ref", "target": "foo-target-4"},
            ],
        },
    )


@pytest.fixture
def ref_mapping():
    yield AttributeMapping(
        ref_type="openminds/AtlasAnnotation",
        refs={
            "ref-id-0": [{}],
            "ref-id-1": [{"target": "foo-target-1"}, {"target": "foo-target-2"}],
            "ref-id-2": [{"target": "foo-target-1"}, {"target": "foo-target-2"}],
            "ref-id-4": [],
            "ref-id-5": [{"target": "foo-target-3"}, {"target": "foo-target-4"}],
        },
    )


def test_filter_by_target_none(region_mapping, ref_mapping):
    assert region_mapping.filter_by_target() == AttributeMapping(
        parcellation_id="foo-bar",
        region_mapping={
            "foo-region-0": [{"@type": "volume/ref"}],
        },
    )

    assert ref_mapping.filter_by_target() == AttributeMapping(
        ref_type="openminds/AtlasAnnotation",
        refs={
            "ref-id-0": [{}],
        },
    )


def test_filter_by_target_valid(region_mapping, ref_mapping):
    assert region_mapping.filter_by_target("foo-target-1") == AttributeMapping(
        parcellation_id="foo-bar",
        region_mapping={
            "foo-region-0": [{"@type": "volume/ref"}],
            "foo-region-1": [
                {"@type": "volume/ref", "target": "foo-target-1"},
            ],
            "foo-region-2": [
                {"@type": "volume/ref", "target": "foo-target-1"},
            ],
        },
    )

    assert ref_mapping.filter_by_target("foo-target-1") == AttributeMapping(
        ref_type="openminds/AtlasAnnotation",
        refs={
            "ref-id-0": [{}],
            "ref-id-1": [{"target": "foo-target-1"}],
            "ref-id-2": [{"target": "foo-target-1"}],
        },
    )


def test_filter_by_target_indvalid(region_mapping, ref_mapping):
    assert region_mapping.filter_by_target("nomatch") == AttributeMapping(
        parcellation_id="foo-bar",
        region_mapping={
            "foo-region-0": [{"@type": "volume/ref"}],
        },
    )

    assert ref_mapping.filter_by_target("nomatch") == AttributeMapping(
        ref_type="openminds/AtlasAnnotation",
        refs={
            "ref-id-0": [{}],
        },
    )
