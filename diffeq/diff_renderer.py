#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2021-03-17
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
from typing import List

import torch
import numpy as np
from pytorch3d.transforms import axis_angle_to_matrix
from torch import Tensor
from trimesh import Trimesh

from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, TexturesVertex, BlendParams, softmax_rgb_blend,
)
from pytorch3d.structures import Meshes
from torch.nn import Parameter as Par


class SimpleShader(torch.nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = softmax_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image


class MeshesRenderer(torch.nn.Module):
    def __init__(self, meshes: List[Trimesh], meshes_colors: List[np.ndarray], cameras,
                 device: str = "cpu", image_size=(240, 240), dtype=torch.float32, perspective_correct=None):
        super().__init__()
        bs = cameras.get_camera_center().shape[0]
        self.batch_size = bs
        self.device = device

        """ Use mesh to create packed representation of all meshes per batch. """
        v_colors = [torch.from_numpy(c).float().unsqueeze(0).repeat(m.vertices.shape[0], 1) for m, c in
                    zip(meshes, meshes_colors)]
        pmeshes = Meshes(
            verts=[torch.from_numpy(m.vertices).float() for m in meshes],
            faces=[torch.from_numpy(m.faces) for m in meshes],
            textures=TexturesVertex(v_colors)
        ).to(device)
        self.vertices = pmeshes.verts_padded().to(device, dtype)
        verts_features_packed = TexturesVertex(v_colors).verts_features_packed().to(device)
        flattened_vertices = self.vertices.unsqueeze(0).expand(bs, len(meshes), -1, 3).reshape(bs, -1, 3)
        self.padded_to_packed_idx = pmeshes.verts_padded_to_packed_idx()
        self.meshes = Meshes(
            verts=flattened_vertices[:, self.padded_to_packed_idx].to(dtype=dtype),
            faces=pmeshes.faces_packed().unsqueeze(0).expand(self.batch_size, -1, 3).to(dtype=dtype),
            textures=TexturesVertex(verts_features_packed.unsqueeze(0).expand(self.batch_size, -1, 3).to(dtype=dtype))
        ).to(device)

        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=RasterizationSettings(
                    image_size=image_size,
                    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
                    faces_per_pixel=16,
                    perspective_correct=perspective_correct,
                )
            ),
            shader=SimpleShader(device=device, blend_params=blend_params)
        ).to(device)

    def forward(self, meshes_positions: Tensor, meshes_rotations: Tensor, meshes_scale: Tensor,
                cam_pos: Tensor, cam_rot: Tensor) -> Tensor:
        """ B x numMeshes x ... """
        scaled_v = meshes_scale.unsqueeze(-2) * self.vertices
        vers = (meshes_rotations.unsqueeze(-3) @ scaled_v.unsqueeze(-1)).squeeze(-1) + meshes_positions.unsqueeze(-2)
        flattened_padded = vers.reshape(self.batch_size, -1, 3)
        images = self.renderer(meshes_world=self.meshes.update_padded(flattened_padded[:, self.padded_to_packed_idx]),
                               R=cam_rot, T=cam_pos)
        return images


class CameraModelTzAndElevation(torch.nn.Module):
    def __init__(self, cam2world_tz=0., elevation_deg=10., device='cpu', dtype=torch.float32):
        super().__init__()
        tparam = dict(device=device, dtype=dtype)
        self.tparam = tparam
        self.cam2world_tz = Par(torch.tensor([cam2world_tz], **tparam))
        self.elevation = Par(torch.tensor([elevation_deg], **tparam).deg2rad())

        self.cam2world_fixed_rot = axis_angle_to_matrix(torch.tensor([0., 0., np.pi], **tparam)) @ \
                                   axis_angle_to_matrix(torch.tensor([np.pi / 2, 0., 0.], **tparam))

        self.cam2world_txy = torch.zeros(2, **tparam)
        self.cam2world_ryz = torch.zeros(2, **tparam)

    @property
    def cam2world_rot(self):
        return self.cam2world_fixed_rot @ axis_angle_to_matrix(torch.cat([self.elevation, self.cam2world_ryz]))

    @property
    def cam2world_t(self):
        return torch.cat([self.cam2world_txy, self.cam2world_tz])

    @property
    def world2cam_rot(self):
        return self.cam2world_rot.T  # R.T

    @property
    def world2cam_t(self):
        return -(self.cam2world_rot.T @ self.cam2world_t.unsqueeze(-1)).squeeze(-1)  # -R.T @ t

    def transform_cam2world(self, pos, rot, detach=False):
        """ Given nx3 position and nx3x3 rotation in camera frame, compute position and rotation in world frame."""
        n = pos.shape[0]
        pos = self.cam2world_rot.unsqueeze(0).expand(n, 3, 3).bmm(pos.unsqueeze(-1)).squeeze(-1) + self.cam2world_t
        rot = self.cam2world_rot.unsqueeze(0).expand(n, 3, 3).bmm(rot)
        if detach:
            return pos.detach(), rot.detach()
        return pos, rot

    def cam_pos_rot(self, n):
        """ Return camera position and rotation in form expected by pytorch3d cameras for batch size n"""
        cam_rot = self.world2cam_rot.T.unsqueeze(0).expand(n, 3, 3)
        cam_pos = self.world2cam_t.unsqueeze(0).expand(n, 3)
        return cam_pos, cam_rot
