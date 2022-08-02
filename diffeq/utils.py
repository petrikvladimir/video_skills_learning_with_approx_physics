#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2022-02-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
from pytorch3d.transforms.so3 import hat
from torch import zeros, eye, Tensor, transpose
from torch.nn import Parameter as Par
from pytorch3d.transforms import matrix_to_rotation_6d, so3_log_map, rotation_6d_to_matrix


def vhat(omega: Tensor) -> Tensor:
    """ Compute hat operator on a single vector [3]. """
    return hat(omega.unsqueeze(0)).squeeze(0)


def init_vec_param(a, device, dtype, init_value=None):
    """ Get initial vector that servers as nn.Parameter's initialization if not none, otherwise use zeros."""
    if init_value is None:
        init_value = zeros(3, device=device, dtype=dtype)
    return Par(a.to(device=device, dtype=dtype) if a is not None else init_value, True)


def init_pos_param(a, device, dtype):
    """ Get initial vector that servers as nn.Parameter's initialization if not none, otherwise use zeros."""
    return init_vec_param(a, device, dtype)


def init_rot_param(a, device, dtype):
    """
    Get initial rot. matrix that servers as nn.Parameter's initialization if not none, otherwise use identity.
    Rot6D representation is used for underlying parameters.
    """
    if a is None:
        return Par(matrix_to_rotation_6d(eye(3, device=device, dtype=dtype).unsqueeze(0)).squeeze(0), True)
    return Par(matrix_to_rotation_6d(a.to(device=device, dtype=dtype).unsqueeze(0)).squeeze(0), True)


def velocities_from_positions(pos, timestamps):
    """ From given positions [n] and corresponding timestamps [n] in seconds, computes velocities [n]"""
    return (pos[1:] - pos[:-1]) / (timestamps[1:] - timestamps[:-1]).unsqueeze(-1)


def velocities_from_rotations(rot, timestamps):
    """ From given positions [n] and corresponding timestamps [n] in seconds, computes velocities [n]"""
    drot = rot[1:].bmm(transpose(rot[:-1], -2, -1))
    return so3_log_map(drot) / (timestamps[1:] - timestamps[:-1]).unsqueeze(-1)


def fix_rot(rot):
    if rot.shape == (3, 3):
        return rotation_6d_to_matrix(matrix_to_rotation_6d(rot.unsqueeze(0))).squeeze(0)
    return rotation_6d_to_matrix(matrix_to_rotation_6d(rot))
