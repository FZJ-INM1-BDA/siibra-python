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

import requests
from typing import TYPE_CHECKING

from . import cite_proc_json
from .base import content_type_registry

if TYPE_CHECKING:
    from ...attributes.descriptions import Doi


def get_citation(doi: "Doi"):
    if len(content_type_registry) == 0:
        raise RuntimeError("No known content type registered.")

    url = doi.url

    headers = {
        "Accept": ", ".join(
            [f"{content_type}" for content_type in content_type_registry]
        )
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type")
    content_type, *_ = content_type.split(";")  # filter out "; charset utf-8"
    assert (
        content_type in content_type_registry
    ), f"Got content type content_type={content_type!r}. This type has not been registered"

    result = content_type_registry[content_type](resp.content)
    return result