#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2021-02-16
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

from typing import Union
from torch import Tensor, norm, cat, tensor
import torch


def sdf_to_ground_plane(points: Tensor, ground_axis: int = 2) -> Tensor:
    """
    Compute signed distance of points [Bx3] to the ground plane that has normal in the axis orthogonal to the
    Cartesian base defined by the ground_axis (z by default). The plane crosses the origin.
    """
    return points[..., ground_axis]


def sdf_to_sphere(points: Tensor, pos: Tensor, rot: Tensor, sizes: Tensor) -> Tensor:
    """ Compute signed distance [B] of points [Bx3] to the sphere given it's pose [3], [3x3] and sizes [1]. """
    return norm(points - pos, dim=-1) - sizes


def sdf_to_box(points: Tensor, pos: Tensor, rot: Tensor, sizes: Tensor) -> Tensor:
    """
    Compute signed distance [B] of points [Bx3] to the box given it's pose [3], [3x3] and sizes [3].
    Note, that returned distance is not euclidean.
    """
    p_in_body_frame = (rot.T @ (points - pos).unsqueeze(-1)).squeeze(-1)
    hl = sizes / 2.
    dist_to_planes = cat([-p_in_body_frame - hl, p_in_body_frame - hl], dim=-1)
    return torch.max(dist_to_planes, dim=-1)[0]


def sdf_to_box_batch(points: Tensor, pos: Tensor, rot: Tensor, sizes: Tensor) -> Tensor:
    """
    Compute signed distance [B] of points [Bx3] to the corresponding boxes given it's pose [Bx3], [Bx3x3]
    and sizes [Bx3]. Note, that returned distance is not euclidean.
    """
    p_in_body_frame = (rot.transpose(-2, -1) @ (points - pos).unsqueeze(-1)).squeeze(-1)
    hl = sizes / 2.
    dist_to_planes = cat([-p_in_body_frame - hl, p_in_body_frame - hl], dim=-1)
    return torch.max(dist_to_planes, dim=-1)[0]


def sdf_to_box_with_normals(points: Tensor, pos: Tensor, rot: Tensor, sizes: Tensor) -> Union[Tensor, Tensor]:
    """
    Compute signed distance [B] of points [Bx3] to the box given it's pose [3], [3x3] and sizes [3].
    Note, that returned distance is not euclidean.
    """
    p_in_body_frame = (rot.T @ (points - pos).unsqueeze(-1)).squeeze(-1)
    hl = sizes / 2.
    dist_to_planes = cat([-p_in_body_frame - hl, p_in_body_frame - hl], dim=-1)
    box_normals = Tensor([
        [-1., 0., 0.],
        [0., -1., 0.],
        [0., 0., -1.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ]).to(pos)
    v, ind = torch.max(dist_to_planes, dim=-1)
    return v, (rot @ box_normals[ind].T).T


def sdf_to_box_with_normals_without_bottom(points: Tensor, pos: Tensor, rot: Tensor, sizes: Tensor) -> Union[
    Tensor, Tensor]:
    """
    Compute signed distance [B] of points [Bx3] to the box given it's pose [3], [3x3] and sizes [3].
    Note, that returned distance is not euclidean.
    """
    sides = tensor([True, True, False, True, True, True]).to(pos.device)
    p_in_body_frame = (rot.T @ (points - pos).unsqueeze(-1)).squeeze(-1)
    hl = sizes / 2.
    dist_to_planes = cat([-p_in_body_frame - hl, p_in_body_frame - hl], dim=-1)
    box_normals = Tensor([
        [-1., 0., 0.],
        [0., -1., 0.],
        [0., 0., -1.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ]).to(pos)[sides]
    v, ind = torch.max(dist_to_planes[:, sides], dim=-1)
    return v, (rot @ box_normals[ind].T).T
