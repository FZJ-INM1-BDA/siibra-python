from .image_fetcher import ImageFetcher, cache_and_load_img


class NeuroglancerFetcher(ImageFetcher, srcformat="neuroglancer/precomputed"):

    def __init__(self, url):
        super().__init__(url)

    def fetch(self):
        return cache_and_load_img(self.url)


class NeuroglancerMeshFetcher(ImageFetcher, srcformat="neuroglancer/precompmesh"):

    def __init__(self, url):
        super().__init__(url)

    def fetch(self):
        return cache_and_load_img(self.url)
