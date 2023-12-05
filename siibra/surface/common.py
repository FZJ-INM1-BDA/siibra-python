from typing import List, Union, IO
import numpy as np
import trimesh
from io import BytesIO
from functools import wraps


class Surface:
    def __init__(
        self,
        vertices: List[List[float]] = None,
        faces: List[List[int]] = None,
        vertex_labels: List[List[Union[float, int]]] = None,
        **kwargs,
    ) -> None:
        """
        Describes a common interface to surface, where vertices
        faces are defined, and optionally, vertices labels, face labels
        can be defined.

        Parameters
        ----------
        vertices (List[List[float]] or ndarray): verticies triplet
        faces: (List[List[int]] or ndarray): faces triplet
        vertex_labels: (Labels)
        """

        self.vertices = np.array(
            kwargs.get("verts", []) if vertices is None else vertices
        )
        self.faces = np.array([] if faces is None else faces, dtype=np.uint64)

        vertex_labels = (
            kwargs.get("labels", []) if vertex_labels is None else vertex_labels
        )
        self.vertex_labels = np.array(vertex_labels)

    def __contains__(self, spec):
        self.faces
        if spec in ("verts", "faces"):
            return True
        if spec == "labels":
            return len(self.vertex_labels) > 0
        return False

    def __getitem__(self, spec: str):
        if spec == "verts":
            return self.vertices
        if spec == "faces":
            return self.faces
        if spec == "labels":
            if len(self.vertex_labels) > 0:
                return self.vertex_labels
        raise IndexError(f"spec {spec!r} not found in class surface")

    def export(self, export_dest: Union[str, IO]):
        mesh = trimesh.Trimesh(self.vertices, self.faces)
        mesh.export(export_dest)

    def to_bytes(self):
        io = BytesIO()
        mesh = trimesh.Trimesh(self.vertices, self.faces)
        mesh.export(io, file_obj="obj")
        io.seek(0)
        return io.read()


def wrap_return_surface():
    def outer(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            return_val = fn(*args, **kwargs)
            return Surface(**return_val)

        return inner

    return outer
