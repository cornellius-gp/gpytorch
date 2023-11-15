#!/usr/bin/env python3

from typing import List

import numpy as np
import torch
from scipy.spatial import Voronoi as scipyVoronoi

from ._index import BaseIndex


def voronoi_finite_polygons_2d(vor: scipyVoronoi, radius: float = None) -> (List[torch.tensor], torch.tensor):
    """
    Authored by @Sklavit, modified by me.

    Reconstruct infinite voronoi regions in a 2D diagram to finite regions. Returns regions in the order of the input
    points, rather than the order of the regions in vor.

    :param vor: scipy.spatial.Voronoi instance
    :param radius: Distance to 'points at infinity'.

    :return regions: List of tensors, where the ith tensor contains the indices of the ith finite Voronoi region.
    :return vertices: Tensor of shape (m,2) containing the coordinates of each vertex of the Voronoi diagram.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    # could probably make this work with tensors rather than lists of tuples
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions. This also reorders by point_region
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return [torch.tensor(new_region) for new_region in new_regions], torch.tensor(new_vertices)


def is_inside(polygon: torch.tensor, points: torch.tensor) -> torch.tensor:
    """
    Ray tracing algorithm to determine which of an array of points is contained within a polygon.

    @param polygon: Tensor where the first n rows correspond to n vertices in the polygon,
        and the final row contains a duplicate of the first vertex to complete the polygon.
    @param points: Tensor of size (m, 2) representing the points of interest.

    @return: Tensor where the ith element is 1 if the ith point in points belongs to the polygon, 0 otherwise.
    """

    length = len(polygon) - 1
    dy2 = points[:, 1] - polygon[0][1]
    intersections = torch.zeros(points.shape[0], dtype=torch.int)
    ii = 0
    jj = 1

    while ii < length:
        dy = dy2
        dy2 = points[:, 1] - polygon[jj][1]

        mask = (dy * dy2 <= 0) & ((points[:, 0] >= polygon[ii][0]) | (points[:, 0] >= polygon[jj][0]))
        f = dy[mask] * (polygon[jj][0] - polygon[ii][0]) / (dy[mask] - dy2[mask]) + polygon[ii][0]
        intersections[mask] = intersections[mask] + (points[mask, 0] > f).int()
        ii = jj
        jj += 1

    return intersections % 2


class VoronoiIndex(BaseIndex):
    """
    This index constructs a Voronoi diagram from a given feature set, computes neighboring blocks, enables
    evaluating block membership for test points, and enables reordering of the blocks based on the inducing points
    used to construct the diagram.

    :param data: Features to use for Voronoi diagram, typically an (n,2) tensor of spatial lat-long coordinates.
    :param n_blocks: Number of desired polygons. Note that this does not guarantee similarly-sized clusters.
    :param n_neighbors: Number of neighboring polygons per polygon.
    :param seed: Seed for randomly selected inducing points from training points.
    """

    def __init__(self, data: torch.tensor, n_blocks: int, n_neighbors: int, distance_metric, seed: int = None):

        self.n_blocks = n_blocks
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric

        self.inducing_points = None
        self.regions = None
        self.vertices = None

        # this call executes set_blocks and set_neighbors, then superclass computes all dependent quantities
        super(VoronoiIndex, self).__init__(set_blocks_kwargs={"data": data, "seed": seed}, set_neighbors_kwargs={})

    def _get_cluster_membership(self, data: torch.tensor) -> List[torch.LongTensor]:
        """
        Determines which Voronoi region each point in the provided data belongs to.

        :param data: Tensor for which to evaluate Voronoi region membership. If any of these points are outside the
            domain of the points that the Voronoi diagram was constructed with, you may get nonsensical results.

        :return: List of tensors, where the ith tensor contains the indices of the points in data that belong to the
            ith Voronoi region.
        """
        blocks = []
        remaining_idx = torch.LongTensor(range(len(data)))

        # due to doing the loop in this way, blocks will be in the same order as self.inducing_points
        for region in self.regions:
            # add first vertex to end of region to complete the polygon, then compute vertices
            this_region = torch.cat((region, region[0].unsqueeze(dim=0)))
            these_vertices = self.vertices[this_region]

            # get indices of remaining points that belong to this polygon and append those indices to list of blocks
            members = is_inside(these_vertices, data[remaining_idx, :]).nonzero().squeeze()
            blocks.append(remaining_idx[members])

        return blocks

    def set_blocks(self, data: torch.tensor, seed: int) -> List[torch.LongTensor]:
        if seed is not None:
            torch.manual_seed(seed)
        # randomly sample points for constructing voronoi diagram and create diagram
        self.inducing_points = data[torch.randperm(len(data))[0 : self.n_blocks]]
        vor = scipyVoronoi(self.inducing_points, incremental=False)

        # pass scipy voronoi through method to make all regions finite
        self.regions, self.vertices = voronoi_finite_polygons_2d(vor)

        # determine indices of data points that belong to each voronoi region and return
        return self._get_cluster_membership(data)

    def set_neighbors(self) -> List[torch.LongTensor]:
        # if there are no neighbors, we want a list of empty tensors
        if self.n_neighbors == 0:
            return [torch.LongTensor([]) for _ in range(0, self.n_blocks)]

        else:
            # get distance matrix and find ordered distances
            sorter = self.distance_metric(self.inducing_points, self.inducing_points).argsort().long()
            return [sorter[i][sorter[i] < i][0 : self.n_neighbors] for i in range(0, len(sorter))]

    def set_test_blocks(self, new_data: torch.tensor) -> List[torch.LongTensor]:
        # determine indices of new data points that belong to each voronoi region and return
        return self._get_cluster_membership(new_data)

    def reorder(self, ordering_strategy):
        # new order is defined as some reordering of the inducing points for the Voronoi diagram
        new_order = ordering_strategy(self.inducing_points)

        # reorder the instance attributes that depend on the ordering
        self.inducing_points = self.inducing_points[new_order]
        self.regions = [self.regions[idx] for idx in new_order]

        # reorder superclass attributes and recompute neighbors under new ordering
        super()._reorder(new_order)
