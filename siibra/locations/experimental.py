# Copyright 2018-2025
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

from typing import List
from math import atan2

import numpy as np
from nilearn import image

from . import point, pointcloud, boundingbox
from ..volumes import volume
from ..commons import translation_matrix, y_rotation_matrix


class AxisAlignedPatch(pointcloud.PointCloud):

    def __init__(self, corners: pointcloud.PointCloud):
        """Construct a patch in physical coordinates.
        As of now, only patches aligned in the y plane of the physical space
        are supported."""
        # TODO: need to ensure that the points are planar, if more than 3
        assert len(corners) == 4
        assert len(np.unique(corners.coordinates[:, 1])) == 1
        pointcloud.PointCloud.__init__(
            self,
            coordinates=corners.coordinates,
            space=corners.space,
            sigma_mm=corners.sigma_mm,
            labels=corners.labels,
        )

    def __str__(self):
        return f"Patch with boundingbox {self.boundingbox}"

    def flip(self):
        """Returns a flipped version of the patch."""
        new_corners = self.coordinates.copy()[[2, 3, 0, 1]]
        return AxisAlignedPatch(pointcloud.PointCloud(new_corners, self.space))

    def extract_volume(
        self,
        image_volume: volume.Volume,
        resolution_mm: float,
    ):
        """
        fetches image data in a planar patch.
        TODO The current implementation only covers patches which are strictly
        define in the y plane. A future implementation should accept arbitrary
        oriented patches.accept arbitrary oriented patches.
        """
        assert image_volume.space == self.space

        # Extend the 2D patch into a 3D structure
        # this is only valid if the patch plane lies within the image canvas.
        canvas = image_volume.get_boundingbox()
        assert canvas.minpoint[1] <= self.coordinates[0, 1]
        assert canvas.maxpoint[1] >= self.coordinates[0, 1]
        XYZ = self.coordinates
        voi = boundingbox.BoundingBox(
            XYZ.min(0)[:3], XYZ.max(0)[:3], space=image_volume.space
        )
        # enforce the patch to have the same y dimensions
        voi.minpoint[1] = canvas.minpoint[1]
        voi.maxpoint[1] = canvas.maxpoint[1]
        patch = image_volume.fetch(voi=voi, resolution_mm=resolution_mm)
        assert patch is not None

        # patch rotation defined in physical space
        vx, vy, vz = XYZ[1] - XYZ[0]
        alpha = -atan2(-vz, -vx)
        cx, cy, cz = XYZ.mean(0)
        rot_phys = np.linalg.multi_dot(
            [
                translation_matrix(cx, cy, cz),
                y_rotation_matrix(alpha),
                translation_matrix(-cx, -cy, -cz),
            ]
        )

        # rotate the patch in physical space
        affine_rot = np.dot(rot_phys, patch.affine)

        # crop in the rotated space
        pixels = np.dot(np.linalg.inv(affine_rot), self.homogeneous.T).astype("int").T
        # keep a pixel distance to avoid black border pixels
        xmin, ymin, zmin = pixels.min(0)[:3] + 1
        xmax, ymax, zmax = pixels.max(0)[:3] - 1
        h, w = xmax - xmin, zmax - zmin
        affine = np.dot(affine_rot, translation_matrix(xmin, 0, zmin))
        return volume.from_nifti(
            image.resample_img(
                patch,
                target_affine=affine,
                target_shape=[h, 1, w],
                force_resample=True
            ),
            space=image_volume.space,
            name=f"Rotated patch with corner points {self.coordinates} sampled from {image_volume.name}",
        )


class Plane:
    """
    A 3D plane in reference space.
    TODO This shall eventually be derived from siibra.Location
    """

    def __init__(self, point1: point.Point, point2: point.Point, point3: point.Point):
        """
        Create a 3D plane from 3 points.
        The plane's reference space is defined by the first point.
        """
        self.space = point1.space
        # normal is the cross product of two arbitrary in-plane vectors
        n = np.cross(
            (point2.warp(self.space) - point1).coordinate,
            (point3.warp(self.space) - point1).coordinate,
        )
        self._n = n / np.linalg.norm(n)
        # distance from origin is the projection of any in-plane point onto the normal
        self._d = np.dot(point1.coordinate, self._n)

    @property
    def normal(self):
        return self._n

    @property
    def distance_from_origin(self):
        return self._d

    def sidedness(self, points: np.ndarray):
        """
        Computes labels for a set of 3D coordinates classifying them
        by the halfspaces spanned by this plane.
        """
        assert points.shape[1] == 3
        return (np.dot(points, self.normal) >= self.distance_from_origin).astype("int")

    def intersect_line_segments(self, startpoints: np.ndarray, endpoints: np.ndarray):
        """
        Intersects a set of straight line segments with the plane.
        Returns the set of intersection points.
        The line segments are given by two Nx3 arrays of their start- and endpoints.
        The result is an Nx3 list of intersection coordinates.
        """
        directions = endpoints - startpoints

        # Use raw parametric t in [0, 1] — no need to normalize.
        # t=0 means startpoint, t=1 means endpoint.
        denom = directions @ self._n
        t = (self._d - startpoints @ self._n) / denom

        result = startpoints + t[:, None] * directions

        non_intersecting = (t < 0) | (t > 1)
        num_failed = np.count_nonzero(non_intersecting)
        result[non_intersecting] = np.nan
        if num_failed > 0:
            print(
                "WARNING: line segment intersection includes NaN rows "
                f"for {num_failed} non-intersecting segments."
            )
        return result

    def intersect_mesh(self, mesh: dict) -> List[pointcloud.Contour]:
        """
        Intersects a 3D surface mesh with the plane.
        Returns a set of split 2D contours, represented by ordered coordinate lists.
        The output contour coordinates are intersection points of mesh edges going through the plane,
        and only combined into a contour if arising from neighboring edges in the mesh.
        The mesh is passed as a dictionary with an Nx3 array "verts" of vertex coordinates,
        and an Mx3 array "faces" of face definitions.
        Each row in the face array corresponds to the three indices of vertices making up the
        triangle.
        The result is a list of contour segments, each represented as a PointCloud
        holding the ordered list of contour points.
        The point labels in each "contour" PointCloud hold the index of the face in the
        mesh which made up each contour point.
        """
        # Select faces whose vertices span both halfspaces relative to the plane.
        vertex_in_halfspace = self.sidedness(mesh["verts"])
        face_vertex_in_halfspace = vertex_in_halfspace[mesh["faces"]]
        face_indices_arr = np.where(
            face_vertex_in_halfspace.min(1) != face_vertex_in_halfspace.max(1)
        )[0]
        faces = mesh["faces"][face_indices_arr]

        # For each selected face, identify forward (1) and backward (-1) plane crossings
        # across edges 2->0, 0->1, 1->2.
        crossings = np.diff(
            face_vertex_in_halfspace[face_indices_arr][:, [2, 0, 1, 2]], axis=1
        )
        face_columns = np.array([[2, 0], [0, 1], [1, 2]])

        # Verify exactly one forward and one backward crossing per face.
        # NOTE: will fail if an edge lies exactly in-plane.
        assert all(all((crossings == v).sum(1) == 1) for v in [-1, 0, 1])

        # Compute forward and backward intersection points.
        fwd_columns = np.where(crossings == 1)[1]
        bwd_columns = np.where(crossings == -1)[1]
        fwd_vertices = np.array(
            [
                faces[np.arange(len(faces)), face_columns[fwd_columns][:, 0]],
                faces[np.arange(len(faces)), face_columns[fwd_columns][:, 1]],
            ]
        ).T
        bwd_vertices = np.array(
            [
                faces[np.arange(len(faces)), face_columns[bwd_columns][:, 1]],
                faces[np.arange(len(faces)), face_columns[bwd_columns][:, 0]],
            ]
        ).T
        fwd_intersections = self.intersect_line_segments(
            mesh["verts"][fwd_vertices[:, 0]], mesh["verts"][fwd_vertices[:, 1]]
        )
        bwd_intersections = self.intersect_line_segments(
            mesh["verts"][bwd_vertices[:, 0]], mesh["verts"][bwd_vertices[:, 1]]
        )

        # Verify fwd and bwd intersection sets are identical (up to ordering).
        sortrows = lambda A: A[np.lexsort(A.T[::-1]), :]
        err = (sortrows(fwd_intersections) - sortrows(bwd_intersections)).sum()
        assert err == 0, f"intersection inconsistency: {err}"

        # --- Contour tracing ---
        # Build a lookup from bwd intersection point -> face index.
        # This replaces the O(N²) np.where/np.isin scan in the original loop.
        bwd_lookup = {tuple(pt): i for i, pt in enumerate(bwd_intersections)}

        # Use a set for O(1) membership checks and removal.
        # Ordering within face_indices only determined which face starts the next
        # contour segment — arbitrary in the original code too (was range(N)).
        unvisited = set(range(fwd_intersections.shape[0]))

        result = []
        while unvisited:
            face_id = next(iter(unvisited))  # pick any unvisited face to start
            points = []
            labels = []

            while True:
                p = fwd_intersections[face_id]
                points.append(p)
                labels.append((face_id, fwd_vertices[face_id, 0], fwd_vertices[face_id, 1]))
                unvisited.discard(face_id)

                face_id = bwd_lookup[tuple(p)]  # O(1) lookup replacing O(N) scan
                if face_id not in unvisited:
                    break  # contour is closed or dead-end

            result.append(
                pointcloud.Contour(np.array(points), labels=labels, space=self.space)
            )

        return result

    def project_points(self, points: pointcloud.PointCloud):
        """projects the given points onto the plane."""
        assert self.space == points.space
        XYZ = points.coordinates
        N = XYZ.shape[0]
        dists = np.dot(self._n, XYZ.T) - self._d
        return pointcloud.PointCloud(
            XYZ - np.tile(self._n, (N, 1)) * dists[:, np.newaxis], space=self.space
        )

    def get_enclosing_patch(self, points: pointcloud.PointCloud, margin=[0.5, 0.5]):
        """
        Computes the enclosing patch in the given plane
        which contains the projections of the given points.
        The orientation of the patch follows the principal axis.
        """
        projections = self.project_points(points)

        # compute PCA of point projections to obtain two orthogonal
        # in-plane spanning vectors
        XYZ = np.copy(projections.coordinates)
        mu = XYZ.mean(0)
        XYZ -= mu
        cov = np.dot(XYZ.T, XYZ)
        eigvals_, eigvecs_ = np.linalg.eigh(cov)
        eigvecs = eigvecs_[:, ::-1].T
        v1, v2 = [-eigvecs[_] for _ in np.argsort(eigvals_)[:2]]

        # get extremal points along first spanning vector
        order = np.argsort(np.dot(projections.coordinates, v1))
        p0 = projections[order[0]].homogeneous[0, :3]
        p1 = projections[order[-1]].homogeneous[0, :3]

        # get extremal points along second spanning vector
        order = np.argsort(np.dot(projections.coordinates, v2))
        p2 = projections[order[0]].homogeneous[0, :3]
        p3 = projections[order[-1]].homogeneous[0, :3]

        m0, m1 = margin
        w = np.linalg.norm(p3 - p2)
        corners = pointcloud.PointCloud(
            [
                p1 + (w / 2 + m1) * v2 + m0 * v1,
                p0 + (w / 2 + m1) * v2 - m0 * v1,
                p0 - (w / 2 + m1) * v2 - m0 * v1,
                p1 - (w / 2 + m1) * v2 + m0 * v1,
            ],
            space=self.space,
        )
        err = (self.project_points(corners).coordinates - corners.coordinates).sum()
        if err > 1e-5:
            print(
                f"WARNING: patch coordinates were not exactly in-plane (error={err})."
            )

        try:
            patch = AxisAlignedPatch(self.project_points(corners))
        except AssertionError:
            patch = None
        return patch

    @classmethod
    def from_image(cls, image: volume.Volume):
        """
        Derive an image plane by assuming the volume to be a 2D image.
        The smallest dimension in voxel space is considered flat.
        The plane is defined in the physical space of the volume.
        """
        assert isinstance(image, volume.Volume)
        im_lowres = image.fetch(resolution_mm=1)
        plane_dims = np.where(np.argsort(im_lowres.shape) < 2)[0]
        voxels = pointcloud.PointCloud(
            np.vstack(([0, 0, 0], np.identity(3)[plane_dims])), space=None
        )
        points = voxels.transform(im_lowres.affine, space=image.space)
        return cls(points[0], points[1], points[2])
