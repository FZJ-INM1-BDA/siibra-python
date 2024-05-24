import requests

from . import cite_proc_json
from .base import content_type_registry
from ...descriptions import Doi


def get_citation(doi: Doi):
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
    assert (
        content_type in content_type_registry
    ), f"Got content type {content_type=!r}. This type has not been registered"

    try:
        result = content_type_registry[content_type](resp.content)
    except Exception as e:
        print("erro", url)
        raise e from e
    return result
