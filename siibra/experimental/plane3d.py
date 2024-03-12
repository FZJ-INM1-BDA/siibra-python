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

from . import contour
from . import patch
from ..locations import point, pointset
from ..volumes import volume

import numpy as np


class Plane3D:
    """
    A 3D plane in reference space.
    This shall eventually be derived from siibra.Location
    """

    def __init__(self, point1: point.Point, point2: point.Point, point3: point.Point):
        """
        Create a 3D plane from 3 points.
        The plane's reference space is defined by the first point.
        """
        self.space = point1.space
        # normal is the cross product of two arbitray in-plane vectors
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
        TODO This returns an intersection even if the line segment intersects the plane,

        """
        directions = endpoints - startpoints
        lengths = np.linalg.norm(directions, axis=1)
        directions = directions / lengths[:, None]
        lambdas = (self._d - np.dot(startpoints, self._n)) / np.dot(directions, self._n)
        assert all(lambdas >= 0)
        result = startpoints + lambdas[:, None] * directions
        non_intersecting = lambdas > lengths
        num_failed = np.count_nonzero(non_intersecting)
        result[non_intersecting, :] = np.nan
        if num_failed > 0:
            print(
                "WARNING: line segment intersection includes NaN rows "
                f"for {num_failed} non-intersecting segments."
            )
        return result

    def intersect_mesh(self, mesh: dict):
        """
        Intersects a 3D surface mesh with the plane.
        Returns a set of split 2D contours, represented by ordered coordinate lists.
        The output contour coordinates are intersection points of mesh edges going through the plane,
        and only combined into a contour if arising from neighboring edges in the mesh.
        The mesh is passed as a dictionary with an Nx3 array "verts" of vertex coordinates,
        and an Mx3 array "faces" of face definitions.
        Each row in the face array corresponds to the three indices of vertices making up the
        triangle.
        The result is a list of contour segments, each represented as a PointSet
        holding the ordered list of contour points.
        The point labels in each "contour" PointSet hold the index of the face in the
        mesh which made up each contour point.
        """

        # select faces whose vertices are in different halfspaces relative to the y plane
        vertex_in_halfspace = self.sidedness(mesh["verts"])
        face_vertex_in_halfspace = vertex_in_halfspace[mesh["faces"]]
        face_indices = np.where(
            face_vertex_in_halfspace.min(1) != face_vertex_in_halfspace.max(1)
        )[0]
        faces = mesh["faces"][face_indices]

        # for each of N selected faces, indicate wether we cross the plane
        # as we go from vertex 2->0, 0->1, 1->2, respectively.
        # This gives us an Nx3 array, where forward crossings are identified by 1,
        # and backward crossings by -1.
        # Each column of the crossings is linked to two columns of the faces array.
        crossings = np.diff(
            face_vertex_in_halfspace[face_indices][:, [2, 0, 1, 2]], axis=1
        )
        face_columns = np.array([[2, 0], [0, 1], [1, 2]])

        # We assume that there is exactly one forward and one inverse crossing
        # per selected face. Test this assumption.
        # NOTE This will fail if an edge is exactly in-plane
        assert all(all((crossings == v).sum(1) == 1) for v in [-1, 0, 1])

        # Compute the actual intersection points for forward and backward crossing edges.
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

        # By construction, the fwd and backward intersections
        # should include the exact same set of points. Verify this now.
        sortrows = lambda A: A[np.lexsort(A.T[::-1]), :]
        err = (sortrows(fwd_intersections) - sortrows(bwd_intersections)).sum()
        assert err == 0

        # Due to the above property, we can construct closed contours in the
        # intersection plane by following the interleaved fwd/bwd roles of intersection
        # points.
        face_indices = list(range(fwd_intersections.shape[0]))
        result = []
        points = []
        labels = []
        face_id = 0  # index of the mesh face to consider
        while len(face_indices) > 0:

            # continue the contour with the next foward edge intersection
            p = fwd_intersections[face_id]
            points.append(p)
            # Remember the ids of the face and start-/end vertices for the point
            labels.append((face_id, fwd_vertices[face_id, 0], fwd_vertices[face_id, 1]))
            face_indices.remove(face_id)
            neighbours = np.where(np.all(np.isin(bwd_intersections, p), axis=1))[0]
            assert len(neighbours) > 0
            face_id = neighbours[0]
            if face_id in face_indices:
                # more points available in the contour
                continue

            # finish the current contour.
            result.append(
                contour.Contour(np.array(points), labels=labels, space=self.space)
            )
            if len(face_indices) > 0:
                # prepare to process another contour segment
                face_id = face_indices[0]
                points = []
                labels = []

        return result

    def project_points(self, points: pointset.PointSet):
        """projects the given points onto the plane."""
        assert self.space == points.space
        XYZ = points.coordinates
        N = XYZ.shape[0]
        dists = np.dot(self._n, XYZ.T) - self._d
        return pointset.PointSet(
            XYZ - np.tile(self._n, (N, 1)) * dists[:, np.newaxis], space=self.space
        )

    def get_enclosing_patch(self, points: pointset.PointSet, margin=[0.5, 0.5]):
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
        corners = pointset.PointSet(
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
            print(f"WARNING: patch coordinates were not exactly in-plane (error={err}).")
        return patch.Patch(self.project_points(corners))

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
        voxels = pointset.PointSet(
            np.vstack(([0, 0, 0], np.identity(3)[plane_dims])), space=None
        )
        points = voxels.transform(im_lowres.affine, space=image.space)
        return cls(points[0], points[1], points[2])
