from dataclasses import dataclass

from .base import Description


@dataclass
class Doi(Description):
    schema = "siibra/attr/desc/doi/v0.1"
    value: str = None

    @property
    def url(self):

        url = self.value
        assert (
            "doi.org" in url
        ), f"doi.value must have 'doi.org' in its value, but {url=} does not."
        if url.startswith("http://"):
            url = "https://" + url.replace("http://", "", 1)
        if not url.startswith("https://"):
            url = "https://" + url
        return url
