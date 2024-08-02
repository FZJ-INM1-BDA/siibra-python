from typing import Iterable
from ebrains_drive import BucketApiClient

from .base import ArchivalRepository

class DataproxyRepository(ArchivalRepository):
    """
    Proxy to BucketApiClient, satisfying ArchivalRepository constraint.
    
    User can access dataset buckets by passing is_bucket=False. n.b. not all dataset
    stores files in dataproxy.
    
    User can provide a JWT token to access private buckets/dataset. n.b. if a token is passed
    the instance of dataproxyrepository will *no longer* be able to access public dataset/bucket
    """

    anony_client = BucketApiClient()

    def __init__(self, bucketname: str, is_bucket: bool=True, token: str=None) -> None:
        super().__init__()
        self.bucketname = bucketname
        self.client = BucketApiClient(token=token) if token else self.anony_client
        if is_bucket:
            self.bucket = self.client.buckets.get_bucket(bucketname)
        else:
            self.bucket = self.client.buckets.get_dataset(bucketname)

    @property
    def unpacked_dir(self):
        return None

    def warmup(self, *args, **kwargs):
        raise NotImplementedError(f"dataproxy repository cannot be warmed up")
    
    def search_files(self, prefix: str = None) -> Iterable[str]:
        for f in self.bucket.ls(prefix):
            yield f.name

    def get(self, filepath: str):
        return self.bucket.get_file(filepath).get_content()
