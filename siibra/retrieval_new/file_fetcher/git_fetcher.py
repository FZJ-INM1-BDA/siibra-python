from typing import Iterable, Dict, List
import requests
from dataclasses import dataclass
import zlib

from .base import ArchivalRepository


HEADERS = {"content-type": "application/x-git-upload-pack-request"}


@dataclass
class TreeResult:
    mode: str
    filename: str
    sha: str


@dataclass
class Commit:
    tree: str
    parent: str
    msg: str


class GitHttpRepository(ArchivalRepository):

    def __init__(self, url: str, branch: str) -> None:
        self.sess = requests.Session()
        self.base_url = url
        self.ref_map: Dict[str, str] = {}
        self.head = None
        self.branch = branch

    def _populate_ref_map(self):

        resp = requests.get(f"{self.base_url}/info/refs")
        resp.raise_for_status()
        for line in resp.text.split("\n"):
            if line:
                digest, ref = line.split()
                self.ref_map[ref] = digest
        assert len(self.ref_map) > 0
        assert (
            f"refs/heads/{self.branch}" in self.ref_map
        ), f"branch {self.branch} is not found."
        self.head = self.ref_map[f"refs/heads/{self.branch}"]

    @property
    def unpacked_dir(self):
        return None

    def decode_tree(self, b: bytes):

        hdr, body = b.split(b"\x00", 1)
        arr: List[TreeResult] = []
        while True:
            meta, rest = body.split(b"\x00", 1)
            mode, filename = meta.decode().split(" ", 1)
            sha = rest[:20].hex()

            arr.append(TreeResult(mode=mode, filename=filename, sha=sha))

            body = rest[20:]
            if len(body) == 0:
                break
        return arr

    def decode_commit(self, b: bytes):
        hdr, body = b.split(b"\x00", 1)
        tree_line, parent_line, *_ = body.decode().split("\n")
        tree_key, treesha = tree_line.split(" ")
        parent_key, parentsha = parent_line.split(" ")
        assert tree_key == "tree"
        assert parent_key == "parent"
        return Commit(tree=treesha, parent=parentsha, msg=body.decode())

    def get_object(self, sha: str):

        resp = self.sess.get(f"{self.base_url}/objects/{sha[:2]}/{sha[2:]}")
        resp.raise_for_status()
        result = zlib.decompress(resp.content)
        return result

    def ls(self):
        if self.head is None:
            self._populate_ref_map()
        assert self.head, "Error: head is not set"
        commit_obj = self.get_object(self.head)
        commit = self.decode_commit(commit_obj)

        tree_obj = self.get_object(commit.tree)
        decoded_tree = self.decode_tree(tree_obj)
        return decoded_tree

    def search_files(self, prefix: str) -> Iterable[str]:
        raise NotImplementedError

    def warmup(self, *args, **kwargs):
        pass

    def get(self, filepath: str):
        raise NotImplementedError
