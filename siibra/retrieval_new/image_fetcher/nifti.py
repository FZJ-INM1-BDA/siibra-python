from .image_fetcher import ImageFetcher, cache_and_load_img


class NiftiFetcher(ImageFetcher, srcformat="nii"):

    def __init__(self, url):
        super().__init__(url)

    def fetch(self):
        return cache_and_load_img(self.url)
