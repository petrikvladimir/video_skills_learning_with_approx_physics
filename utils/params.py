#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2022-01-27
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

import argparse
import numpy as np
from pytorch3d.renderer import PerspectiveCameras
from trimesh import creation as trimesh_creation


def get_default_arg_parser():
    parser = argparse.ArgumentParser(description='Differentiable physics skills learning.')
    parser.add_argument('vid', type=str)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-cuda_id', type=int, default=0)
    parser.add_argument('-subsample_time', type=int, default=1)
    parser.add_argument('-subsample_img', type=int, default=1)
    return parser


def get_perspective_camera(device, image_size, batch_size):
    """ Create perspective cameras used for given image_size and batch_size"""
    fov_x, fov_y = np.deg2rad(80), np.deg2rad(60)
    fx = image_size[1] / (2 * np.tan(fov_x / 2))
    fy = image_size[0] / (2 * np.tan(fov_y / 2))
    camera = PerspectiveCameras(
        device=device, image_size=[image_size] * batch_size,
        focal_length=[(fx, fy)] * batch_size, principal_point=[(image_size[1] / 2, image_size[0] / 2)] * batch_size,
        in_ndc=False,
    )
    return camera


def get_cylinder_hand_mesh():
    """ Return hand mesh represented as a cylinder """
    return trimesh_creation.cylinder(height=0.4, radius=0.04).apply_translation(np.array([0., 0., 0.2]))
