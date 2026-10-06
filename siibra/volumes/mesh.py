from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import trimesh


def from_mesh_dict(
    data: Mapping[str, Any], *, process: bool = False
) -> trimesh.Trimesh:
    vertices = data.get("vertices", data.get("verts"))
    faces = data.get("faces")

    if vertices is None:
        raise KeyError("Mesh data must contain 'vertices' or 'verts'.")
    if faces is None:
        raise KeyError("Mesh data must contain 'faces'.")

    labels = data.get("labels")

    vertex_attributes = {}
    face_attributes = {}

    if labels is not None:
        labels = np.asarray(labels)

        if len(labels) == len(vertices):
            vertex_attributes["labels"] = labels
        elif len(labels) == len(faces):
            face_attributes["labels"] = labels
        else:
            raise ValueError(
                "labels must match either number of vertices or number of faces."
            )

    metadata = {
        key: value
        for key, value in data.items()
        if key not in {"vertices", "verts", "faces", "labels"}
    }

    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=process,
        metadata=metadata,
        vertex_attributes=vertex_attributes,
        face_attributes=face_attributes,
    )
