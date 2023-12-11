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

from typing import Union
from .feature import Feature
get = Feature._match


TYPES = Feature._get_subclasses()  # Feature types that can be used to query for features


def __dir__():
    return list(Feature._CATEGORIZED.keys()) + ["get", "TYPES", "render_ascii_tree"]


def __getattr__(attr: str):
    if attr in Feature._CATEGORIZED:
        return Feature._CATEGORIZED[attr]
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
        _ = ftype._get_instances()


def render_ascii_tree(class_or_classname: Union[type, str]):
    """
    Print the ascii hierarchy representation of a feature type.

    Parameters
    ----------
    Cls: Feature class, str
        Any Feature class or string of the feature type name
    """
    from anytree.importer import DictImporter
    from anytree import RenderTree
    Cls = TYPES[class_or_classname] if isinstance(class_or_classname, str) else class_or_classname

    def create_treenode(feature_type):
        return {
            'name': feature_type.__name__,
            'children': [
                create_treenode(c)
                for c in feature_type.__subclasses__()
            ]
        }
    D = create_treenode(Cls)
    importer = DictImporter()
    tree = importer.import_(D)
    print("\n".join(
        "{}{}".format(pre, node.name)
        for pre, _, node in RenderTree(tree)
    ))
