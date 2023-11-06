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
"""Multimodal data features types and query mechanisms."""

from . import (
    connectivity,
    tabular,
    image,
    dataset,
)


from .feature import Feature
get = Feature.match


TYPES = Feature._get_subclasses()


def __dir__():
    return list(Feature.CATEGORIZED.keys())


def __getattr__(attr: str):
    if attr in Feature.CATEGORIZED:
        return Feature.CATEGORIZED[attr]
    else:
        hint = ""
        if isinstance(attr, str):
            import difflib
            closest = difflib.get_close_matches(attr, list(__dir__()), n=3)
            if len(closest) > 0:
                hint = f"Did you mean {' or '.join(closest)}?"
        raise AttributeError(f"No such attribute: {__name__}.{attr} " + hint)


def warm_cache():
    """Preload preconfigured multimodal data features."""
    for ftype in TYPES.values():
        _ = ftype.get_instances()
