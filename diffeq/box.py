#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2021-03-3
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import itertools

import torch
from torch import Tensor
from typing import Tuple
from trimesh.creation import box as trimesh_box

from diffeq.distances import sdf_to_box, sdf_to_ground_plane, sdf_to_box_with_normals, \
    sdf_to_box_with_normals_without_bottom


class Box(torch.nn.Module):

    def __init__(self, num_points_edge=5, device='cpu', dtype=torch.float64) -> None:
        """ Create a box with given initial size [3] and specifies if size is used in backpropagation. """
        super().__init__()
        self.unit_mesh = trimesh_box(extents=[1.] * 3)  # mesh is needed for visualization purposes
        # self.unit_q_b = torch.from_numpy(self.mesh.vertices).float().to(device=device, dtype=dtype)  # n x 3
        span = torch.linspace(-0.5, 0.5, num_points_edge)[1:-1]
        self.unit_q_b = torch.tensor(
            [[0.5, i, j] for i, j in itertools.product(span, span)] +
            [[-0.5, i, j] for i, j in itertools.product(span, span)] +
            [[i, 0.5, j] for i, j in itertools.product(span, span)] +
            [[i, -0.5, j] for i, j in itertools.product(span, span)] +
            [[i, j, 0.5] for i, j in itertools.product(span, span)] +
            [[i, j, -0.5] for i, j in itertools.product(span, span)] +
            [[-0.5000, -0.5000, -0.5000], [-0.5000, -0.5000, 0.5000], [-0.5000, 0.5000, -0.5000],
             [-0.5000, 0.5000, 0.5000], [0.5000, -0.5000, -0.5000], [0.5000, -0.5000, 0.5000],
             [0.5000, 0.5000, -0.5000], [0.5000, 0.5000, 0.5000]]
        ).to(device=device, dtype=dtype)

    @staticmethod
    def unit_trimesh_box():
        return trimesh_box(extents=[1.] * 3)

    def q_b(self, size) -> Tensor:
        """ Returns point positions in a body frame properly scaled according to sizes of the box. """
        return self.unit_q_b * size

    def q_s(self, rot, size) -> Tensor:
        """ Return points position in a spatial frame. """
        return (rot @ self.q_b(size).unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def obj_not_grasped_signal(pos: Tensor, rot: Tensor, size: Tensor, hpos: Tensor, hg: Tensor) -> Tensor:
        """ This signal is negative iff box at position pose (pos, rot) is grasped by hand (hx, hg) """
        return torch.max(sdf_to_box(points=hpos, pos=pos, rot=rot, sizes=size), -hg)

    @staticmethod
    def is_grasped(pos: Tensor, rot: Tensor, size: Tensor, hpos: Tensor, hg: Tensor) -> Tensor:
        """ Return 1 if object at pose (pos, rot) is grasped by hand (hx, hg) and -1 otherwise. """
        # return Box.obj_not_grasped_signal(pos=pos, rot=rot, size=size, hpos=hpos, hg=hg).sign().squeeze()
        return torch.tensor(
            [1. if Box.obj_not_grasped_signal(pos=pos, rot=rot, size=size, hpos=hpos, hg=hg) < 0 else -1.]
        ).to(pos)

    @staticmethod
    def points_positions_and_velocities(q_s: Tensor, pos: Tensor, vel: Tensor, omega: Tensor) -> Tuple[Tensor, Tensor]:
        """ For given point positions in spatial frame q_s [Nx3] and pos [3], linear [3] and angular [3] velocities
        of the body return the corresponding points positions and velocities in spatial frame. """
        points_pos = q_s + pos
        points_vel = omega.unsqueeze(0).expand(q_s.shape[0], -1).cross(q_s, dim=-1) + vel.unsqueeze(0)
        return points_pos, points_vel

    def impulse_by_plane_collision(self, pos: Tensor, vel: Tensor, size: Tensor, rot: Tensor, omega: Tensor, kp=5000,
                                   kv=50,
                                   tangential_velocity_reduction=0.) -> Tuple[Tensor, Tensor]:
        """
        Return change of linear and angular velocities of the box caused by the ground plane collision.
        In addition, a tangential friction can be approximated by velocity reduction in time.
        """
        q_s = self.q_s(rot=rot, size=size)
        points_pos, points_vel = self.points_positions_and_velocities(q_s=q_s, pos=pos, vel=vel, omega=omega)

        n = torch.tensor([0., 0., 1.]).to(pos).unsqueeze(0)
        sdf = sdf_to_ground_plane(points=points_pos, ground_axis=2)
        vel = (points_vel * n).sum(-1).clamp_max(0.)  # [N] only colliding velocity in normal direction
        mag = torch.clamp_min(-kp * sdf - kv * vel, 0.)  # [N]
        impulses = mag.unsqueeze(-1) * n  # [N x 3]

        if tangential_velocity_reduction > 0.:
            v_xy = points_vel * torch.tensor([[1, 1, 0]]).to(pos)
            impulses = impulses - tangential_velocity_reduction * mag.sign().unsqueeze(-1) * v_xy

        dvel = impulses.sum(axis=0)
        domega = q_s.cross(impulses).sum(dim=0)
        return dvel, domega

    def impulse_by_static_box_collision(self, pos: Tensor, vel: Tensor, size: Tensor, rot: Tensor, omega: Tensor,
                                        other_box_pos: Tensor, other_box_rot: Tensor,
                                        other_box_size: Tensor, kp=10000, kv=100,
                                        tangential_velocity_reduction=0.
                                        ) -> Tuple[Tensor, Tensor]:
        """
        Return change of linear and angular velocities of the box caused by the collision with other box.
        In addition, a tangential friction can be approximated by velocity reduction in time.
        """
        q_s = self.q_s(rot=rot, size=size)
        points_pos, points_vel = self.points_positions_and_velocities(q_s=q_s, pos=pos, vel=vel, omega=omega)

        sdf, n = sdf_to_box_with_normals_without_bottom(points=points_pos, pos=other_box_pos, rot=other_box_rot,
                                                        sizes=other_box_size)
        vel = (points_vel * n).sum(-1).clamp_max(0.)  # [N] only colliding velocity in normal direction
        mag = torch.clamp_min(-kp * sdf - kv * vel, 0.)  # [N]
        impulses = mag.unsqueeze(-1) * n  # [N x 3]

        if tangential_velocity_reduction > 0.:
            points_vel_in_normal_dir = ((points_vel * n).sum(dim=-1, keepdim=True) * n).to(pos)
            v_xy = points_vel - points_vel_in_normal_dir
            impulses = impulses - tangential_velocity_reduction * mag.sign().unsqueeze(-1) * v_xy

        dvel = impulses.sum(axis=0)
        domega = q_s.cross(impulses).sum(dim=0)
        return dvel, domega
